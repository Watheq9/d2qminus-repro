# Doc2Query-- Reproducibility

This repo contains the code and data used in our reproducibility study of [doc2query--](https://github.com/terrierteam/pyterrier_doc2query) technique.



## Libraries Installation
Before replicating the work, you need to create a conda environment, activate it, and then install the needed libraries. Run the following commands:

```
conda create --name doc2query  -c pytorch -c conda-forge -c nvidia python=3.8.17 pytorch==2.0.1 pytorch-cuda=11.8 faiss-gpu=1.6.5  trec_eval=9.0.8 openjdk=11.0.22
conda activate doc2query
pip install -r requirements.txt
pip install git+https://github.com/terrierteam/pyterrier_dr.git
```

## 1. Scoring Queries
To generate the ELECTRA scores of any dataset, you need to prepare a .jsonl file that contains three columns ['id', 'text', 'predicted_queries']. Then, you can run this command:

```
python experiments/score_generator.py \
--input path_to_your_input_file.jsonl \
--output path_to_save_the_output_file.jsonl \
--log path_to_save_the_log
```


## 2. End-to-end Retrieval using Doc2Query-- (Top or Bottom)
Having scored the queries, you can apply filtering. The following script will apply filtering, index the filtered dataframe, tune BM25 parameters, and then evaluate BM25 on four test collection of MS-MARCO. For example, if you want to experiment with top 10, 30, 50, 70, 90%, run the following command:

```
python experiments/filter_tune_and_test.py --scored_file "path_to_the_scored_file.jsonl" \
    --index_dir "directory_to_save_the_resulted_indexes" \
    --eval_dir "directory_to_save_evaluation_files" \
    --runs_dir "directory_to_save_retrieval_runs" \
    --filter_type 'top' \
    --percentages 10 30 50 70 90
```
To make filtering on bottom side (queries with scores below X threshold), just set "--filter_type 'bottom'"

## 3. Zero-shot retrieval on BEIR
To test Doc2Query-- on BEIR benchmark, you need to generate expansion queries for each dataset and then score the expansion queries. Here you can find 20 expansion queries for each dataset. Download them, format and score the corpus files as described in Step 1. Following this, you can run this command to perform the zero-shot test on DBPedia dataset:

```
python experiments/zero_shot.py --scored_file "path_to_the_dataset_scored_file.jsonl" \
    --index_dir "directory_to_save_the_resulted_indexes" \
    --eval_dir "directory_to_save_evaluation_files" \
    --runs_dir "directory_to_save_retrieval_runs" \
    --filter_type 'top' \
    --percentages 30 50 \
    --dataset 'dbpedia' \
    --pt_name 'irds:beir/dbpedia-entity/test' 
```
Given the zero-shot setup, we only test on two thresholds (30 and 50) as explained in the paper.
It is worth noting that other datasets need providing more arguments, like specifying the query column. 
Here, we show the argument differences for the experimented datasets in the paper.

Robust04 
```
--dataset 'robust04'  --pt_name 'irds:disks45/nocr/trec-robust-2004' --query_column 'description'
```
TREC-COVID 
```
--dataset 'trec-covid'  --pt_name 'irds:beir/trec-covid' --query_column 'text'
```
Quora 
```
--dataset 'quora'  --pt_name 'irds:beir/quora/test' 
```
Touche2020 
```
--dataset 'touche2020'  --pt_name 'irds:beir/webis-touche2020/v2' --query_column 'text'
```

## 4. RL Experiments
Please see the `experiments/rl/` directory for more information.

## 5. LSR Experiments
Please see the `experiments/lsr/` directory for more information.

## 6. Reranking Experiment
Please see the `experiments/raranking/` directory for more information.


## Utility 
To generate filtered file out of a scored file (without the scores or evaluating the filtered file) of any dataset, you can run the following command:

```
python experiments/filter.py --scored_file "path_to_the_scored_file.jsonl" \
                        --filtered_file "path_to_save_the_filtered_file" \
                        --filter_type 'top' \
                        --percentages 30 \
                        --N 80 \
```

