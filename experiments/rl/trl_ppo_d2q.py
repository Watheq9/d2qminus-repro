import torch
from tqdm import tqdm
import numpy as np
from typing import List
import warnings
import argparse
tqdm.pandas()

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, \
    T5ForConditionalGeneration, T5Tokenizer, AutoModel
from datasets import load_dataset, Dataset
import random
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, set_seed
from trl.core import LengthSampler
# from pyserini.analysis import Analyzer, get_lucene_analyzer


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def expansion_reward(passage, query, analyzer):
    passage_words = set(analyzer.analyze(str(passage)))
    query_words = set(analyzer.analyze(str(query)))
    expansion_words = query_words.difference(passage_words)
    if len(query_words) == 0:
        return 0
    return len(expansion_words) / len(query_words)


def build_dataset(config, dataset_name, qrels_file, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir='cache')

    def tokenize(sample):
        passage = sample['text'].strip()
        # text = f"Generate a relevant question for the following passage:\nPassage: {passage}"
        text = passage
        sample["input_ids"] = tokenizer.encode(text, truncation=True, max_length=max_length)
        sample["query"] = passage
        return sample

    ds = load_dataset(dataset_name, cache_dir='cache')

    # filter out passages that are in qrels
    qrels = set()
    with open(qrels_file) as f:
        for line in f:
            qid, _, pid, _ = line.split()
            qrels.add(pid)
    ds = ds.filter(lambda x: x['docid'] not in qrels)

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def main():
    ########################################################################
    # This is a fully working simple example to use trl with accelerate.
    #
    # This example fine-tunes a GPT2 model on the IMDB dataset using PPO
    # (proximal policy optimization).
    # in any of the following settings (with the same script):
    #   - single CPU or single GPU
    #   - multi GPUS (using PyTorch distributed mode)
    #   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
    #   - fp16 (mixed-precision) or fp32 (normal precision)
    #
    # To run it in each of these various modes, first initialize the accelerate
    # configuration with `accelerate config`
    #
    ########################################################################


    parser = argparse.ArgumentParser(description="Script to train DocT5Query model using Reinforcement learning")
    parser.add_argument("--train_qrels", type=str, required=True, help="Path to the train qrels file")
    args = parser.parse_args()
    train_qrels = args.train_qrels

    # We first define the configuration of the experiment, defining the model, the dataset,
    # the training parameters, and the PPO parameters.
    # Check the default arguments in the `PPOConfig` class for more details.
    # If you want to log with tensorboard, add the kwarg
    # `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
    config = PPOConfig(
        model_name='castorini/doc2query-t5-base-msmarco',
        # model_name='google/flan-t5-base',
        learning_rate=1e-5,
        batch_size=64,
        mini_batch_size=64,
        log_with="wandb",
        tracker_project_name="DIL",
        accelerator_kwargs={'mixed_precision': 'fp16'},
        seed=929,
        # use_score_scaling=True,
        # use_score_norm=True,
        ratio_threshold=5,
        adap_kl_ctrl=True,
        kl_penalty='kl',
        init_kl_coef=0.5
    )
    set_seed(config.seed)
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name, cache_dir='cache')
    ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(config.model_name, cache_dir='cache')
    tokenizer = T5Tokenizer.from_pretrained(config.model_name, cache_dir='cache')
    save_dir = 'd2q_monoELECTRA_checkpoints'
    # analyzer = Analyzer(
    #     get_lucene_analyzer())  # default is English analyzer (stemmer: str='porter', stopwords: bool=True)

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.


    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,  # don't ignore the EOS token (see above)
        "top_k": 0.0,  # no top-k sampling
        "top_p": 1.0,  # no nucleus sampling
        "do_sample": True,  # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id,  # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 32,  # specify how many tokens you want to generate at most
    }
    # output_min_length = 4
    # output_max_length = 32
    # output_length_sampler = LengthSampler(output_min_length, output_max_length)


    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.

    dataset = build_dataset(config,
                            dataset_name='Tevatron/msmarco-passage-corpus',
                            qrels_file=train_qrels,
                            max_length=256
                            )['train']

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    device = ppo_trainer.accelerator.device
    # reward_model = AutoModelForSequenceClassification.from_pretrained('castorini/monobert-large-msmarco',
    #                                                                   cache_dir='cache').to(device)
    # reward_tokenizer = AutoTokenizer.from_pretrained('castorini/monobert-large-msmarco', cache_dir='cache')
    reward_model = AutoModelForSequenceClassification.from_pretrained('crystina-z/monoELECTRA_LCE_nneg31', cache_dir='cache').to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator', cache_dir='cache')
    reward_model.eval()

    current_step = 0
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        #### Get response from doc2query model
        response_tensors = []
        cleaned_query_tensors = []
        cleaned_queries = []
        for i, query in enumerate(query_tensors):
            query = query.to(device)
            # gen_len = output_length_sampler()
            response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
            if len(response) <= 2:
                response = torch.tensor([0, 29, 9, 1], dtype=torch.long, device=device)  # set "na" to query
            response_tensors.append(response.squeeze())
            cleaned_query_tensors.append(query)
            cleaned_queries.append(batch["query"][i])

        batch['response'] = []
        for r in response_tensors:
            try:
                batch['response'].append(tokenizer.decode(r, skip_special_tokens=True))
            except:
                batch['response'].append("na")

        batch["query"] = cleaned_queries

        #### Compute sentiment score
        features = reward_tokenizer(batch["query"], batch["response"], padding=True, truncation=True,
                                    return_tensors="pt").to(device)

        with torch.no_grad():
            output, = reward_model(**features, return_dict=False)
            # scores = torch.nn.functional.softmax(
            #     output / 0.8, 1)[:, 1] * 2
            scores = torch.nn.functional.sigmoid(output[:, 1] - 4)
            # scores = output[:, 1]
            batch['logit'] = output[:, 1].cpu().detach().tolist()
        rewards = [s.cpu().detach() for s in scores]
        # expansion_rewards = [torch.tensor(expansion_reward(passage, query, analyzer)) for passage, query in zip(batch["query"], batch["response"])]
        # final_rewards = [r1 + r2 for r1, r2 in zip(rewards, expansion_rewards)]
        final_rewards = rewards

        #### Run PPO step
        assert len(cleaned_query_tensors) == len(response_tensors) == len(final_rewards)
        ppo_trainer.config.batch_size = len(cleaned_query_tensors)
        stats = ppo_trainer.step(cleaned_query_tensors, response_tensors, final_rewards)
        current_step += 1
        stats["env/reward_model_mean"] = np.mean(rewards)
        stats["env/reward_model_std"] = np.std(rewards)
        # stats["env/reward_expansion_mean"] = np.mean(expansion_rewards)
        # stats["env/reward_expansion_std"] = np.std(expansion_rewards)

        if current_step % 200 == 0:
            ppo_trainer.log_stats(stats, batch, final_rewards, columns_to_log=["query", "response", "logit"])
            ppo_trainer.save_pretrained(f"{save_dir}/checkpoint-{current_step}")
        else:
            ppo_trainer.log_stats(stats, batch, final_rewards, columns_to_log=["query", "response", "logit"])


        # for q, p, r in zip(batch["response"], batch["query"], rewards):
        #     print(f"Epoch:{epoch} Reward:{r} \n Generated query:{q} \n passage: {p}")
        # print()

if __name__ == "__main__":
    main()