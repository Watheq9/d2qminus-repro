import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from pyterrier.measures import *
import helper.evaluation as evaluation
import helper.util as util
import argparse
import pyterrier as pt
from pathlib import Path
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')




def process(scored_file, index_dir, eval_dir, runs_dir, N = 80, 
            filter_type = "top", percentages = [10, 30, 50, 70, 90], 
            measure = RR@10, dev = "irds:msmarco-passage/dev/small"):
    '''
    scored_file: the file which contains the corpus, expansion queries with their corresponding relevance scores
    index_dir: Directory to save indexes within
    eval_dir: Directory to save the evaluation results and logs
    runs_dir: Directory to save the retrieval runs
    N: number of expansion queries in the input file
    filter_type: whether to keep top or bottom expansion queries
    percentages: a list of percentages to experiment with. For example, if filter_type is top and the
                the percentages are [10, 30, 50, 70, 90], then experiment with top_10, top_30, ... , top_90
    measure: the tuning measure for BM25
    dev: the name of dev set in Pyterrier format
    '''
    
    percetages_text = '-'.join(map(str, percentages))
    log_file = os.path.join(eval_dir, f"{filter_type}-{percetages_text}.log")
    all_eval_file = os.path.join(eval_dir, F"EVALUATION_OF_{filter_type}-{percetages_text}.xlsx")
    logger = util.get_logger(log_file)

    if filter_type == 'top':
        filter_function = util.keep_top
    elif filter_type == 'bottom':
        filter_function = util.keep_bottom
 
    df_ans = pd.DataFrame()
    df_scored = None

    for percentage in percentages:

        if filter_type == 'top':
            # keep scores higher than keep_percentage percentile
            # For example, to keep top 30, we need to set the threshold to the 70 percentile and 
            # set the fitler function to keep_high
            keep_percentage = 100 - percentage 

        else: # filter_type == 'bottom' 
            keep_percentage = percentage

        logger.info(f"Processing {filter_type} {percentage}")
        index_path = f"{index_dir}/{filter_type}_{percentage}_from_first_{N}_pisa_index_one_field"
        tuning_save_file  = f"{eval_dir}/{filter_type}_{percentage}_tuning_b_and_k.csv"
        eval_save_file = f"{eval_dir}/{filter_type}_{percentage}_N-{N}-test_tuned_BM25.csv"
        run_initial_name = f"BM25_{filter_type}-{percentage}-N-{N}"
        tuning_runs_dir = f"{runs_dir}/tuning_{filter_type}_{percentage}"

        
        # 1. Build the index/es
        if not util.is_index_built(index_path):
            if df_scored is None: # read input file and compute thresholds for once only
                df_scored = pd.read_json(scored_file, lines=True, dtype={'id': str,}) 
                percentiles = util.compute_percentiles(df_scored['querygen_score'].values)

            threshold = percentiles[f'p_{keep_percentage}']
            df_filtered = util.get_filtered_data(df=df_scored, threshold=threshold, 
                                            filter_function=filter_function, add_text=True)
            logger.info(f"Done preparing the filtered dataframe")
            logger.info(f"Building the index on this path : {index_path}")
            util.build_index(df_filtered, index_path)
            logger.info(f"Done Indexing")

        # 2. Tune BM25 parameters
        if Path(tuning_save_file).exists(): # already tuned BM25
            best_b, best_k = evaluation.get_best_b_and_k(tuning_save_file)
        else: 
            # tune b and k
            logger.info(f"Tuning BM25 parameters for {filter_type} {percentage}")
            best_b, best_k, df_res_b_k, best_score = evaluation.tune_b_and_k1(index_path=index_path, dev=dev, 
                                                                        measure=measure, runs_dir=tuning_runs_dir,
                                                                        save_path=tuning_save_file)
        
        logger.info(f"For {filter_type} {percentage}, best_b = {best_b}, best_k = {best_k}")
        
        # 3. Test BM25 on test sets
        df_res = evaluation.test_BM25_and_save_runs(index_path, best_b, best_k, runs_dir, 
                                        run_initial_name=run_initial_name,
                                        eval_save_file=eval_save_file)

        df_res['N'] = [N] * len(df_res)
        df_res[f'{filter_type}'] = [percentage] * len(df_res)
        logger.info(df_res.to_string())
        df_ans = pd.concat([df_ans, df_res], ignore_index=True)
        df_ans.to_excel(all_eval_file, index=False)




def main():

    parser = argparse.ArgumentParser(description="Script to apply and evalaute filtering on expansion queries of a corpus")
    parser.add_argument("--scored_file", type=str, required=True, help="Path to the input scored queries file")
    parser.add_argument("--index_dir", type=str, required=True, help="Directory to save indexes in")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory to save the evaluation results and logs")
    parser.add_argument("--filter_type", type=str, required=True, help="Can be one of two values only (top or bottom)")
    parser.add_argument("--runs_dir", type=str, required=True, help="Directory to save the retrieval runs")
    parser.add_argument("--percentages", metavar='N', type=int, nargs='+', required=True, help="a list of percentages to experiment with")
    args = parser.parse_args()

    args = parser.parse_args()
    scored_file = args.scored_file
    index_dir = args.index_dir
    eval_dir = args.eval_dir
    runs_dir = args.runs_dir
    filter_type = args.filter_type
    percentages = args.percentages

    if filter_type != 'top' and filter_type != 'bottom':
        raise ValueError('Filter type value must be either top or bottom')

    process(scored_file, index_dir, eval_dir, runs_dir, N = 80, 
            filter_type = filter_type, percentages = percentages, 
            measure = RR@10, dev = "irds:msmarco-passage/dev/small")



if __name__ == "__main__":
    main()