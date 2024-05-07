
import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
from pyterrier.measures import *
import helper.evaluation as evaluation
import helper.preprocessing as preprocessing
import helper.util as util
import pyterrier as pt
import argparse
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')


def build_index(df_input=None, index_path=None,  logger=None):
    
    if not util.is_index_built(index_path):
        logger.info(f"Building the index on this path : {index_path}")
        util.build_index(df_input, index_path)
        logger.info(f"Done Indexing")


def evaluate(index_path, logger, runs_dir, run_initial_name, eval_save_file, dataset=None, 
             k1=0.9, b=0.4, query_column=None, qrels=None, queries=None):
    logger.info(f"Testing BM25 with b = {b}, k1 = {k1}, on index  = {index_path}")
    df_res = evaluation.test_BM25_and_save_runs(index_path, b, k1, runs_dir, 
                                run_initial_name=run_initial_name,
                                eval_save_file=eval_save_file, 
                                dataset=dataset, query_column=query_column, 
                                qrels=qrels, queries=queries)

    logger.info(df_res.to_string())
    return df_res


def get_threshold(df_scored, percentage, filter_type='top'):
    if filter_type == 'top':
        # keep scores higher than keep_percentage percentile
        # For example, to keep top 30, we need to set the threshold to the 70 percentile and 
        # set the fitler function to keep_high
        keep_percentage = 100 - percentage 

    else: # filter_type == 'bottom' 
        keep_percentage = percentage
    percentiles = util.compute_percentiles(df_scored['querygen_score'].values)

    return percentiles[f'p_{keep_percentage}']


def index_evaluate(index_dir, eval_dir, runs_dir, dataset, pt_dataset_name, scored_file, 
                   query_column=None, qrels=None, queries=None, N = 20, filter_type='top',
                   percentages=[30, 50]):
    
    df_ans = pd.DataFrame()
    all_eval_file = f'{eval_dir}/all_evaluation.xlsx'
    exp_runs_dir = f'{runs_dir}/{dataset}'
    log_file = f'{eval_dir}/index_evaluate.log'

    text_only_exp = 'text_only' # text only experiment
    d2q_exp = f'd2q_N{N}'
    text_index = f"{index_dir}/{dataset}_{text_only_exp}"
    d2q_index = f"{index_dir}/{dataset}_{d2q_exp}"

    logger = util.get_logger(log_file)

    if not util.is_index_built(text_index):
        df_scored = pd.read_json(scored_file, lines=True)
        df_text = util.get_original_text(df_scored)
        df_d2q = util.prepare_d2q_for_indexing(df_scored)
        build_index(df_text, text_index, logger)
        build_index(df_d2q, d2q_index, logger)

        # build doc2query-- indexes
        filter_function = util.keep_top if filter_type == 'top' else util.keep_bottom
        for percentage in percentages:
            # compute the thresholds for  top X
            filter_name = f'd2q_N{N}_{filter_type}_{percentage}' # f (Filtered)
            filtered_index_path = f"{index_dir}/{dataset}_{filter_name}"
            threshold = get_threshold(df_scored, percentage=percentage, filter_type=filter_type)
            df_filtered = util.get_filtered_data(df=df_scored, threshold=threshold, 
                                            filter_function=filter_function, add_text=True)
            build_index(df_filtered, filtered_index_path, logger)
            

    # 1. Evaluate text only
    df_res = evaluate(text_index, logger, exp_runs_dir, run_initial_name=text_only_exp, 
                    eval_save_file=f'{eval_dir}/{text_only_exp}.csv', 
                    dataset=pt_dataset_name, query_column=query_column, qrels=qrels, queries=queries)
    
    df_res['experiment'] = [text_only_exp] * len(df_res)
    df_ans = pd.concat([df_ans, df_res], ignore_index=True)

    # 2. Evaluate doc2query
    df_res = evaluate(d2q_index, logger, exp_runs_dir, run_initial_name=d2q_exp, 
                    eval_save_file=f'{eval_dir}/{d2q_exp}.csv', 
                    dataset=pt_dataset_name, query_column=query_column, qrels=qrels, queries=queries)
    
    df_res['experiment'] = [d2q_exp] * len(df_res)
    df_ans = pd.concat([df_ans, df_res], ignore_index=True)

    # 3. Evaluate doc2query-- (on multiple  thresholds)
    for percentage in percentages:
        # compute the thresholds for  top X
        filter_name = f'd2q_N{N}_{filter_type}_{percentage}' # f (Filtered)
        filtered_index_path = f"{index_dir}/{dataset}_{filter_name}"
        eval_file  = f'{eval_dir}/{filter_name}.csv'
        df_res = evaluate(filtered_index_path, logger, exp_runs_dir, run_initial_name=filter_name, 
                        eval_save_file=eval_file, dataset=pt_dataset_name, query_column=query_column, 
                        qrels=qrels, queries=queries)
        
        df_res['experiment'] = [filter_name] * len(df_res)
        df_ans = pd.concat([df_ans, df_res], ignore_index=True)

    # save the whole evaluation file 
    df_ans.to_excel(all_eval_file, index=False)







def main():

    parser = argparse.ArgumentParser(description="Script to apply and evalaute filtering on expansion queries of a corpus")
    parser.add_argument("--scored_file", type=str, required=True, help="Path to the input scored queries file")
    parser.add_argument("--index_dir", type=str, required=True, help="Directory to save indexes in")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory to save the evaluation results and logs")
    parser.add_argument("--runs_dir", type=str, required=True, help="Directory to save the retrieval runs")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset name to experiment with")
    # optional arguments. Required to set either: 1. qrels + queries, or 2. pt_name
    parser.add_argument("--qrels",  default=None, type=str, required=False, help="Path to qrels file. No need to set pt_name if qrels and queries are provided")
    parser.add_argument("--queries",  default=None, type=str, required=False, help="Path to queries file. No need to set pt_name if qrels and queries are provided")
    parser.add_argument("--pt_name", default=None, type=str, required=False, help="The dataset name in pyterrier format. This argument is required if qrels and queris are not provided")
    parser.add_argument("--filter_type",  default="top", type=str, required=False, help="Can be one of two values only (top or bottom)")
    parser.add_argument("--query_column",  default=None, type=str, required=False, help="The column of the query to be used for retrieval in case there are multiple columns in the topics")
    parser.add_argument("--N",  default=20, type=int, required=False, help="Number of expansion queries per document in the scored file")
    parser.add_argument("--percentages", metavar='N', type=int, nargs='+', default=[30, 50], required=False, help="a list of percentages to experiment with")
    args = parser.parse_args()
    scored_file = args.scored_file
    index_dir = args.index_dir
    eval_dir = args.eval_dir
    runs_dir = args.runs_dir
    dataset = args.dataset
    pt_name = args.pt_name
    filter_type = args.filter_type
    percentages = args.percentages
    query_column = args.query_column
    qrels = args.qrels
    queries = args.queries
    N = args.N

    if filter_type != 'top' and filter_type != 'bottom':
        raise ValueError('Filter type value must be either top or bottom')

    if pt_name is None:
        if qrels is None or queries is None:
            raise ValueError('You have to provide either (qrels+queries) or pt_name (the name of the dataset in the pyterrier format)')
    
    if qrels is not None:
        qrels = pd.read_csv(qrels, names=['qid', 'Q0', 'docno', 'label'], sep='\s+', dtype={"qid": "str", "docno": "str"})
    
    if queries is not None:
        queries = pd.read_csv(queries, names=['qid', 'query'], sep='\t', dtype={"qid": "str", "query": "str"})
        queries['query'] = queries['query'].apply(preprocessing.preprocess)

    index_evaluate(index_dir, eval_dir, runs_dir, dataset, pt_dataset_name=pt_name, 
                   scored_file=scored_file, query_column=query_column, qrels=qrels, N = N,
                   queries=queries, filter_type=filter_type, percentages=percentages)



if __name__ == "__main__":
    main()