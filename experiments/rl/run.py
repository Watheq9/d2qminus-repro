from data import IndexingTrainDataset, GenerateDataset, IndexingCollator, QueryEvalCollator
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
)
from trainer import DSITrainer, DocTqueryTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
import json
import jsonlines
from tqdm import tqdm



@dataclass
class RunArguments:
    tokenizer_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    max_length: Optional[int] = field(default=32)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    prefix: str = field(default='id')
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()

    # We use wandb logger: https://wandb.ai/site.
    # if training_args.local_rank == 0 or training_args.local_rank == -1:  # only on main process
    #     # Initialize wandb run
    #     wandb.login()
    #     wandb.init(project="DIL", name=training_args.run_name)
    # #

    tokenizer = T5Tokenizer.from_pretrained(run_args.tokenizer_name, cache_dir='cache')
    fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.tokenizer_name, cache_dir='cache')
    if run_args.model_path:
        model = T5ForConditionalGeneration.from_pretrained(run_args.model_path)
    else:
        # model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
        raise ValueError("Please specify a model path")



    generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                       max_length=run_args.max_length,
                                       cache_dir='cache',
                                       tokenizer=tokenizer)

    trainer = DocTqueryTrainer(
        do_generation=True,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=QueryEvalCollator(
            tokenizer,
            padding='longest',
        ),
    )
    predict_results = trainer.predict(generate_dataset,
                                      top_k=run_args.top_k,
                                      num_return_sequences=run_args.num_return_sequences,
                                      max_length=run_args.q_max_length)

    results = []
    for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                        desc="Writing file"):
        for tokens, docid in zip(batch_tokens, batch_ids):
            query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(f'{docid.item()}\t{query}\n')
    for i in range(run_args.num_return_sequences):
        with open(f"{training_args.output_dir}/{run_args.prefix}_{i:02d}.tsv", 'w', encoding='utf-8') as f:
            f.writelines(results[i::run_args.num_return_sequences])



if __name__ == "__main__":
    main()

