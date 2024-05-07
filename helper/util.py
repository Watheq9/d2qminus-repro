import numpy as np
import pandas as pd
import os, psutil, json, logging
import helper.evaluation as evaluation
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
import pyterrier as pt
import configure as cf
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')


def get_all_n80_thresholds():
    return {
            'top_95': -3.843,
            'top_90': -2.25,
            'top_85': -1.2883,
            'top_80': -0.5877,
            'top_75': -0.0291,
            'top_70': 0.4419,
            'top_65': 0.8564,
            'top_60': 1.2325,
            'top_55': 1.5829,
            'top_50': 1.9162,
            'top_45': 2.2385,
            'top_40': 2.5547,
            'top_35': 2.8702,
            'top_30': 3.1902,
            'top_25': 3.5226,
            'top_20': 3.8785,
            'top_15': 4.2785,
            'top_10': 4.7547,
            'top_5': 5.3554
            }


def get_msmarco_index_dir():
    return cf.MSMARCO_INDEX_DIR 

def get_index_dir():
    return cf.INDEX_DIR 
        
def get_runs_dir():
    return cf.RUNS_DIR


def get_msmarco_test_sets_and_measures():
    return  [
            ('irds:msmarco-passage/dev/small', RR@10),
            ('irds:msmarco-passage/dev/2', RR@10),
            ('irds:msmarco-passage/trec-dl-2019/judged', nDCG@10),
            ('irds:msmarco-passage/trec-dl-2020/judged', nDCG@10),
        ]

def get_msmarco_test_sets():
    return ['irds:msmarco-passage/dev/small',
            'irds:msmarco-passage/dev/2',
            'irds:msmarco-passage/trec-dl-2019/judged',
            'irds:msmarco-passage/trec-dl-2020/judged']

            
def initailize_logger(logger, log_file, level):
    if not len(logger.handlers):  # avoid creating more than one handler
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


def write_new_line_to_jsonl(file_path, new_line):
    
    with open(file_path, 'a') as file:
        json.dump(new_line, file)  # Append the new JSON object to the JSONL file
        file.write('\n')    

def is_index_built(index_path):
    pisa_index = PisaIndex(index_path, stops='none')
    is_built  = pisa_index.built()
    return is_built


def build_index(df, index_path):
    pisa_index = PisaIndex(index_path, stops='none', overwrite=True)
    index_ref = pisa_index.index(df.to_dict(orient="records"))
    # del df
    return pisa_index


def select_first_n(queries, n=10):
    return queries[:n]


def save_file(df, save_file):
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    
    df.to_csv(save_file, index=False)


def join_text_and_queries(text, queries):
    return text + ' \n ' + ' \n '.join(queries) 

def join_two_texts(text1, text2):
    return text1 + ' \n ' + text2

def get_filtered_data(df, threshold, filter_function, add_text=False, exp_query_column='predicted_queries'):
    # Apply the custom function to 'column1' while passing 'column2' values as a parameter

    df_filtered = pd.DataFrame()
    df_filtered['docno'] = df['id'].astype('str')
    if add_text:
        df_filtered['text'] = df.apply(
            lambda row: filter_function(row[exp_query_column], row['querygen_score'], threshold, row['text']), axis=1)
    else:
        df_filtered[exp_query_column] = df.apply(
            lambda row: filter_function(row[exp_query_column], row['querygen_score'], threshold), axis=1)
        df_filtered['text'] = df['text'].astype('str')
    return df_filtered


def compute_percentiles(query_scores_column):
    percentiles = {}
    query_scores = np.concatenate(query_scores_column)
    ps = np.arange(5, 100, 5).tolist()
    for p in ps:
        percentiles.update({f"p_{p}": round(np.percentile(query_scores, p), 4)})
    return percentiles


def keep_local_top(queries, scores, top, text=""):
    '''
    Keep top queries on a local scale, i.e., per document not over the whole corpus
    '''
    # if top = 10, then we need to compute the 90 percentile and keep queries above this value
    # i.e., compute (100 - top) percentile is the target threshold value
    threshold = round(np.percentile(scores, 100 - top), 3)
    # print(threshold)
    if text != "": # means make text + filtered queries 
        return text + ' \n ' + ' \n '.join(np.array(queries)[np.array(scores) >= threshold].tolist()) 
    else: # just returns the filtered queries 
        return ' \n '.join(np.array(queries)[np.array(scores) >= threshold].tolist())


def keep_bottom(queries, scores, threshold, text=""):
    if text != "": # means make text + filtered queries 
        return text + ' \n ' + ' \n '.join(np.array(queries)[np.array(scores) < threshold].tolist()) 
    else: # just returns the filtered queries 
        return ' '.join(np.array(queries)[np.array(scores) < threshold].tolist())


def keep_top(queries, scores, threshold, text=""):
    if text != "": # means make text + filtered queries 
        return text + ' \n ' + ' \n '.join(np.array(queries)[np.array(scores) >= threshold].tolist()) 
    else: # just returns the filtered queries 
        return ' '.join(np.array(queries)[np.array(scores) >= threshold].tolist())


def prepare_d2q_for_indexing(df):
    df_res = pd.DataFrame()
    df_res['docno'] = df['id'].astype('str')
    df_res['text'] = df.apply(lambda row: join_text_and_queries(row['text'], row['predicted_queries']), axis=1)
    return df_res


def get_original_text(df):
    df_res = pd.DataFrame()
    df_res['docno'] = df['id'].astype('str')
    df_res['text'] = df['text'].astype('str')
    return df_res


def print_memory_usage_MB(logger):
    used_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    logger.info(f"Current usage of memory {used_mb} MB")
    logger.info(f'RAM memory {psutil.virtual_memory()[2]}% used:')
    # Getting usage of virtual_memory in GB ( 4th field)
    logger.info(f'RAM Used (GB): {psutil.virtual_memory()[3]/1000000000}')


def read_jsonl_lines(file_path, n=10000):
    json_objects = []
    # Read the first n lines from the JSONL file
    with open(file_path, 'r') as file:
        for i in range(n):
            line = file.readline()
            if not line:
                break  # Break if the end of the file is reached
            json_objects.append(json.loads(line.strip()))

    # Create a DataFrame from the list of JSON objects
    df = pd.DataFrame(json_objects)
    return df



def prepare_queries(query_path):
    df = pd.read_json(query_path, lines=True, dtype={"_id": "str", "text": "str"})
    new_column_names = {'_id': 'qid', 'text': 'query',}
    df = df.rename(columns=new_column_names)
    df = df[['qid', 'query']]
    return df

def prepare_qrels(qrels_path):
    df = pd.read_csv(qrels_path, sep='\t', dtype={"query-id": "str", "corpus-id": "str"})
    new_column_names = {'query-id': 'qid', 'corpus-id': 'docno', 'score': 'label'}
    df = df.rename(columns=new_column_names)
    return df



def save_dataframe(df, save_file, start=0):
    batch_size = 10000
    length = len(df)
    while start < length:
        df_batch = df[start:min(start+batch_size, length)]
        with open(save_file, 'a') as file:
            for _, row in df_batch.iterrows():
                json_line = row.to_json()  # Convert the row to JSON
                file.write(json_line + '\n')  # Write the JSON line to the file
        start += batch_size


def get_terms(queries, split_word=""):
    '''
    queries should be string, but if a list was passed it will be converted to a joined strig first
    '''
    if len(queries) == 0:
        return []

    if type(queries) is list:
        queries = ' '.join(queries)
        
    q_terms = queries.split() # convert the queries to terms
    q_terms_arr = np.array(list(filter(None, q_terms))) # remove empty strings
    return  q_terms_arr


def get_new_terms(doc_unique_terms, queries, return_joined=False):
    
    q_terms_arr = get_terms(queries)
    if len(q_terms_arr) == 0:
        return []
    res_terms = q_terms_arr[np.isin(q_terms_arr, np.array(doc_unique_terms), invert=True)].tolist()
    if return_joined:
        return " \n ".join(res_terms)
    else:
        return res_terms
    
def get_repeated_terms(doc_unique_terms, queries, return_joined=False):
    
    q_terms_arr = get_terms(queries)
    if len(q_terms_arr) == 0:
        return []
    res_terms = q_terms_arr[np.isin(q_terms_arr, np.array(doc_unique_terms))].tolist()
    if return_joined:
        return " \n ".join(res_terms)
    else:
        return res_terms
