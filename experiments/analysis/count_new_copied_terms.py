
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import os
import argparse
import helper.util as util
import gc
import pandas as pd

    
def get_num_new_repeated_terms(df_inp, doc_column='doc_unique_processed_terms', queries_column='queries_processed'):
    df_terms = pd.DataFrame()
    df_terms['q_repeated_terms'] = df_inp.apply(lambda row: util.get_repeated_terms(row[doc_column],
                                                           row[queries_column]), axis=1, )
    df_terms['q_new_terms'] = df_inp.apply(lambda row: util.get_new_terms(row[doc_column],
                                                           row[queries_column]), axis=1, )

    new_terms_lens = [len(x) for x in df_terms['q_new_terms']]
    repeated_terms_lens = [len(x) for x in df_terms['q_repeated_terms']]

    num_repeated_terms = sum(repeated_terms_lens)
    num_new_terms = sum(new_terms_lens)
    total_query_terms = num_repeated_terms + num_new_terms
    # free memory space
    del df_terms
    gc.collect()
    return total_query_terms, num_new_terms, num_repeated_terms


def process(scored_file, save_file, filter_type, percentage,):
    '''
    scored_file: the file which contains the corpus, expansion queries with their corresponding relevance scores
    save_file: Path to save the results
    filter_type: whether to keep top or bottom expansion queries
    percentage: the percentage of queries to retain over the whole corpus
                For example, if filter_type is top and the the percentage is 30, then retain queries with scrores higher than the 70 percentile
    '''
    

    df_scored = None
    log_file = f"{scored_file}-count_new_copied-of-{filter_type}-{percentage}.log"
    logger = util.get_logger(log_file)
    if filter_type == 'top':
        filter_function = util.keep_top
        # keep scores higher than keep_percentage percentile
        # For example, to keep top 30, we need to set the threshold to the 70 percentile and 
        # set the fitler function to keep_high
        keep_percentage = 100 - percentage 
    elif filter_type == 'bottom':
        filter_function = util.keep_bottom
        keep_percentage = percentage


    logger.info(f"Reading the input file : {scored_file}")
    df_scored = pd.read_json(scored_file, lines=True, dtype={'id': str,}) 
    logger.info(f"Done reading")
    percentiles = util.compute_percentiles(df_scored['querygen_score'].values)
    threshold = percentiles[f'p_{keep_percentage}']
    logger.info(f"Applying filtering of the {filter_type} {percentage}")
    if percentage == 100: # no need to filter
        df_scored['filtered_query_procesed_terms'] = df_scored['queries_processed'].apply(' '.join)
    else:
        df_scored['filtered_query_procesed_terms'] = df_scored.apply(lambda row: filter_function(row['queries_processed'], 
                                                                                    row['querygen_score'], threshold), axis=1)
    
    logger.info(f"Done filtering. Counting new and copied terms for {filter_type} {percentage}")
    total_query_terms, num_new_terms, num_repeated_terms = get_num_new_repeated_terms(df_inp=df_scored, 
                                                                                    doc_column='doc_unique_processed_terms', 
                                                                                    queries_column='filtered_query_procesed_terms')
    row = { f"{filter_type}": percentage,
            "num_query_terms": total_query_terms, 
            "num_new_terms": num_new_terms, 
            "num_repeated_terms": num_repeated_terms, 
            "new_percentage": num_new_terms/total_query_terms * 100 , 
            "repeated_percentage": num_repeated_terms/total_query_terms * 100}

    df_res = pd.DataFrame([row])
    df_res.to_excel(save_file, index=False)

    logger.info(f"Done counting, and the results were saved to {save_file}")
    logger.info(df_res.to_string())
    return df_res



def main():

    parser = argparse.ArgumentParser(description="Script to apply and evalaute filtering on expansion queries of a corpus")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the input scored and processed file")
    parser.add_argument("--save_file", type=str, required=True, help="Path to save the results")
    parser.add_argument("--filter_type", type=str, required=True, help="Can be one of two values only (top or bottom)")
    parser.add_argument("--percentage", type=int, required=True, help="The percentage of filtering. Set 100 for no filtering")
    args = parser.parse_args()
    scored_file = args.processed_file
    save_file = args.save_file
    filter_type = args.filter_type
    percentage = args.percentage

    if filter_type != 'top' and filter_type != 'bottom':
        raise ValueError('Filter type value must be either top or bottom')

    process(scored_file, save_file, filter_type, percentage,)


if __name__ == "__main__":
    main()


