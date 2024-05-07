# Example Re-ranking using DL19
RUN=/path/to/dl19/run/file.trec
OUT=re-ranked.trec


# Re-rank both k=100 and k=1000
for k in 100 1000; do


CUDA_VISIBLE_DEVICES=0 python3 llm-rankers/run.py \
  run --model_name_or_path castorini/monot5-base-msmarco \
      --tokenizer_name_or_path castorini/monot5-base-msmarco \
      --run_path  $RUN \
      --save_path $OUT \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits $k \
      --query_length 32 \
      --passage_length 128 \
      --device cuda \
  pointwise --method yes_no \
            --batch_size 32

done

