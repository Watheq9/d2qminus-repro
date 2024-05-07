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


def process(scored_file, filtered_files, filter_type="top", percentages=[30], N=80):
    '''
    scored_file: the file which contains the corpus, expansion queries with their corresponding relevance scores
    filtered_file: Path to save the filtered file
    N: number of expansion queries in the input file
    filter_type: whether to keep top or bottom expansion queries
    percentages: the percentages of queries to retain over the whole corpus
                For example, if filter_type is top and the the percentage is 30, then retain queries with scrores higher than the 70 percentile
    '''
    

    if filter_type == 'top':
        filter_function = util.keep_top
    elif filter_type == 'bottom':
        filter_function = util.keep_bottom
 
    # read input file and compute thresholds for once only
    df_scored = pd.read_json(scored_file, lines=True, dtype={'id': str,}) 
    percentiles = util.compute_percentiles(df_scored['querygen_score'].values)

    for percentage, filtered_file in zip(percentages, filtered_files):

        log_file = f"{filtered_file}.log"
        logger = util.get_logger(log_file)

        if filter_type == 'top':
            # keep scores higher than keep_percentage percentile
            # For example, to keep top 30, we need to set the threshold to the 70 percentile and 
            # set the fitler function to keep_high
            keep_percentage = 100 - percentage 

        else: # filter_type == 'bottom' 
            keep_percentage = percentage

        logger.info(f"Processing {filter_type} {percentage}")
        threshold = percentiles[f'p_{keep_percentage}']
        df_filtered = util.get_filtered_data(df=df_scored, threshold=threshold, 
                                        filter_function=filter_function, add_text=False)
        logger.info(f"Done preparing the filtered dataframe")
        logger.info(f"Writing the filtered dataframe to this file : {filtered_file}")
        util.save_dataframe(df_filtered, filtered_file)
        logger.info(f"Done writing")

     

def main():

    parser = argparse.ArgumentParser(description="Script to apply and evalaute filtering on expansion queries of a corpus")
    parser.add_argument("--scored_file", type=str, required=True, help="Path to the input scored queries file")
    parser.add_argument("--filtered_file", type=str, required=True, help="Path to save the filtered file(s)")
    parser.add_argument("--filter_type", type=str, required=True, help="Can be one of two values only (top or bottom)")
    parser.add_argument("--percentage", type=int, required=True, help="a list of percentages to filter")
    parser.add_argument("--N",  default=80, type=int, required=False, help="Number of expansion queries per document in the scored file")
    args = parser.parse_args()
    scored_file = args.scored_file
    filtered_files = args.filtered_file
    filter_type = args.filter_type
    percentages = args.percentage
    N = args.N

    if filter_type != 'top' and filter_type != 'bottom':
        raise ValueError('Filter type value must be either top or bottom')

    if len(filtered_files) != len(percentages):
        raise ValueError('Length of filtered_file and  percentages should match')
    
    process(scored_file, filtered_files, filter_type, percentages, N)



if __name__ == "__main__":
    main()