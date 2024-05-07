# Reinforcement Learning Experiments

## 1. Train RL model
First step is to train a Doc2Query model that optimizes the ELECTRA score directly using reinforcement learning. We release our model after training on Huggingface [here](https://huggingface.co/Watheq/d2q_monoELECTRA_1400).
To log training details, you need to create an account on Wandb. Follow the instructions [here](https://docs.wandb.ai/quickstart) for creating the account and do the login. Then, you can run the following script:
```
python experiments/rl/trl_ppo_d2q.py
```
### 2. Generate the expansion queries 
First, you need to download collection.tsv from [here](https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz).
Then, select Doc2Query model trained by RL at the target checkpoint (the one with highest reward), and employ it to generate the expansion queries using the following generation script:

```
export seed=42
export d2q_model=path_to_the_model_checkpoint
torchrun experiments/rl/run.py \
        --task generation \
        --tokenizer_name castorini/doc2query-t5-base-msmarco \
        --model_path ${d2q_model} \
        --per_device_eval_batch_size 128 \
        --run_name docTquery-msmarco-generation-${seed} \
        --max_length 128 \
        --seed ${seed} \
        --top_k 10 \
        --valid_file path_to_the_downloaded_collection.tsv \
        --output_dir RL_generated_queries_dir \
        --dataloader_num_workers 10 \
        --report_to wandb \
        --logging_steps 100 \
        --num_return_sequences 20 \
        --prefix predicted_queries_seed_${seed}
```

