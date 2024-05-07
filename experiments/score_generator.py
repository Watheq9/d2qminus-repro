import numpy as np
import pyterrier as pt
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')
import pandas as pd
import os
import json
from pyterrier_dr import ElectraScorer
import pandas as pd
import logging
import argparse

def initailize_logger(logger, log_file, level):
    
    if not len(logger.handlers): # avoid creating more than one handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

    return logger

def get_logger(log_file="progress.txt", level=logging.DEBUG):

    logger = logging.getLogger(log_file)
    logger = initailize_logger(logger, log_file, level)
    return logger



class QueryScorer(pt.Transformer):
    def __init__(self, scorer):
        self.scorer = scorer

    def transform(self, inp):
          slices = []
          scorer_inp = {
              'query': [],
              'text': [],
          }
          for text, querygen in zip(inp['text'], inp['predicted_queries']):
            #   queries = querygen.split('\n')
              queries = querygen
              start_idx = len(scorer_inp['query'])
              slices.append(slice(start_idx, start_idx+len(queries)))
              scorer_inp['query'].extend(queries)
              scorer_inp['text'].extend([text] * len(queries))
          scorer_inp['qid'] = list(range(len(scorer_inp['query'])))
          dout = self.scorer(pd.DataFrame(scorer_inp))
          return inp.assign(querygen_score=[dout['score'].values[s] for s in slices])


def process(input_file, output_file, logger, start=0):
    '''
    start: the index of the row to start scoring from
    '''

    scorer = QueryScorer(ElectraScorer('crystina-z/monoELECTRA_LCE_nneg31'))
    df_col = pd.read_json(input_file, lines=True, dtype={'id': str,}, )
    total_steps = len(df_col)
    batch_size = 256
    batch = []
    coll_size = len(df_col)
    while start < coll_size:
        batch = df_col[start:min(start+batch_size, coll_size)]
        df_batch = scorer(batch)
        with open(output_file, 'a') as file:
            for _, row in df_batch.iterrows():
                json_line = row.to_json()  # Convert the row to JSON
                file.write(json_line + '\n')  # Write the JSON line to the file

        start  += len(df_batch)
        logger.info(f'Processed {start} documents out of {total_steps}')



    logger.info('Processing all documents has been completed.')




def main():

    parser = argparse.ArgumentParser(description="Script to score expansion queries")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data file")
    parser.add_argument("--log", type=str, required=True, help="Path to the log file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--start", type=int, default=0, required=False, help="the index of the row to start scoring from")
    args = parser.parse_args()
    log_file = args.log
    input = args.input
    output = args.output
    start = int(args.start)
    
    logger = get_logger(log_file)
    logger.info(f"The log file is saved into : {log_file}")
    logger.info(f"The output file is saved into : {output}")
    logger.info(f"The input file is : {input}")
    logger.info(f"Scoring documents is starting from row = {start}")

    process(input, output, logger, start)


if __name__ == "__main__":
    main()