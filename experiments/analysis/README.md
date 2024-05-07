# Analysis Experiments

The code provided below allows partial reproduction of our analysis experiments.

## 1. Preprocessing
To perform analysis for the new and copied terms, it is required to perform preprocessing on the original text and expansion queries, and then keep the unique terms of the original text. 
To achieve so,  you can run the following command:

```
python experiments/analysis/preproces_corpus.py --input "path_to_the_input_scored_file.jsonl" \
                               --output "path_to_save_the_processed_file.jsonl"
```


## 2. Compute new and copied terms percentages
Having preprocessed the corpus with its expansion queries, you can compute how many new and copied terms are present in the expansion queries. You can do that with or without filtering (just set the percentage parameter to 100). To achieve that, run this script:
```
python experiments/analysis/count_new_copied_terms.py \
    --processed_file "path_to_the_processed_file.jsonl" \
    --save_file "path_to_save_results.xlsx" \
    --filter_type "top" \
    --percentage 30
```


## 3. New and copied terms analysis across buckets
We also provide a script to compute proportion of new and copied terms across fixed-range (20% for example) buckets. Similar to the previous script, you need to provide a file that contains the processed terms for the documents and expansion queries along with their scores. You can specify the range_step as needed. For example, if you set it to 20, then we have five buckets ((0%, 20%), (20%, 40%), ... (80%, 100%)). We provide an option to save the buckets as well. To do so, run the following script:

```
python experiments/analysis/analyse_ranges.py \
    --processed_file "path_to_the_processed_file.jsonl" \
    --save_file "path_to_save_results.xlsx" \
    --range_step 20 \
    --buckets_dir "dirctory_to_save_the_buckets_in"
```