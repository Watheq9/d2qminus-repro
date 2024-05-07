
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import os
import argparse
import helper.util as util
import gc
import pandas as pd


MIN_SCORE = -100
MAX_SCORE = 100

def get_terms(queries, split_word=""):
    if len(queries) == 0:
        return []
    # if split_word != "":
    #     queries = " ".join(queries.split(split_word))
    # else: 
    #     queries = " ".join(queries)
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


    
def get_num_new_repeated_terms(df_inp, doc_column='doc_unique_processed_terms', queries_column='queries_processed'):
    df_terms = pd.DataFrame()
    df_terms['q_repeated_terms'] = df_inp.apply(lambda row: get_repeated_terms(row[doc_column],
                                                           row[queries_column]), axis=1, )
    df_terms['q_new_terms'] = df_inp.apply(lambda row: get_new_terms(row[doc_column],
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


def keep_in_range(queries, scores, start, end):
    scores = scores[:len(queries)]
    return np.array(queries)[(np.array(scores) >= start) & (np.array(scores) < end)].tolist() 


def get_starts_ends(queries_scores, range_step):

    ps = np.arange(range_step, 100, range_step).tolist()

    start_percentage = []
    end_percentage = []
    str_values = [MIN_SCORE]
    end_values = []
    for p in ps:
        if len(str_values) != 10:
            str_values.append(round(np.percentile(queries_scores, p), 3))
        end_values.append(round(np.percentile(queries_scores, p), 3))
        start_percentage.append(p-range_step)
        end_percentage.append(p)

    start_percentage.append(100-range_step)
    end_percentage.append(100)
    end_values.append(MAX_SCORE)
    assert len(start_percentage) == len(end_values)
    assert len(end_percentage) == len(str_values)
    return start_percentage, str_values, end_percentage, end_values

def analyse_new_and_repeated_in_ranges(df, save_file, logger,
                             buckets_dir=None, range_step=20,
                             processed_queries_column='queries_processed',
                             processed_docs_column='doc_unique_processed_terms',
                             scores_column='querygen_score',
                             ):
    
    total_number_of_queries  = sum([len(x) for x in df[processed_queries_column]])
    queries_scores = np.concatenate(df[scores_column].values)
    queries_counter = {"name": "num_of_queries"}
    queries_percentage = {"name": "kept queries %"}
    new_terms_sum = {"name": "number of new terms"}
    repeated_terms_sum = {"name": "number of repeated terms"}
    new_terms_average = {"name": "Average of new terms per query"}
    repeated_terms_average = {"name": "Average of repeated terms per query"}
    new_terms_percentage = {"name": "new_terms_percentage"}
    repeated_terms_percentage = {"name": "repeated_terms_percentage"}

    
    start_percentage, str_values, end_percentage, end_values = get_starts_ends(queries_scores, range_step)
  
    for st_per, st_val, en_per, en_val in zip(start_percentage, str_values, end_percentage, end_values):
        logger.info(f"processing [{st_per}, {en_per}]")
        df[f'[{st_per},{en_per}]_processed'] = df.apply(lambda row: keep_in_range(row[processed_queries_column], 
                                                                row[scores_column], start=st_val, end=en_val), axis=1)
        
        
        # save the processed terms in this range
        if buckets_dir is not None: # save two columns (id, the processed terms)
            output_file = f'{buckets_dir}/processed_terms_of_[{st_per}, {en_per}].jsonl'
            df[['id', f'[{st_per},{en_per}]_processed',]].to_json(output_file, orient='records', lines=True)
            logger.info(f"new and repeated terms of [{st_per}, {en_per}] were saved to {output_file} ")

        
        logger.info(f"Done with selecting queries from the range")

        df[f'[{st_per},{en_per}]_repeated'] = df.apply(lambda row: util.get_repeated_terms(row[processed_docs_column], 
                                                                row[f'[{st_per},{en_per}]_processed']), axis=1)
        
        df[f'[{st_per},{en_per}]_new'] = df.apply(lambda row: util.get_new_terms(row[processed_docs_column], 
                                                                row[f'[{st_per},{en_per}]_processed']), axis=1)
        logger.info(f"Done with extracting the new and repeated terms")
        # count number of queries in this range
        num_of_queries = sum([len(x) for x in df[f'[{st_per},{en_per}]_processed']])
        queries_counter.update({f'[{st_per},{en_per}]': num_of_queries})
        queries_percentage.update({f'[{st_per},{en_per}]': num_of_queries/total_number_of_queries})
        
        num_of_new = sum([len(x) for x in df[f'[{st_per},{en_per}]_new']])
        num_of_repeated = sum([len(x) for x in df[f'[{st_per},{en_per}]_repeated']])
        total_num_terms = num_of_new + num_of_repeated

        if num_of_queries == 0:
            num_of_queries = 1
        if total_num_terms == 0:
            total_num_terms = 1

        new_terms_sum.update({f'[{st_per},{en_per}]': num_of_new})
        new_terms_average.update({f'[{st_per},{en_per}]': num_of_new/num_of_queries})
        new_terms_percentage.update({f'[{st_per},{en_per}]': num_of_new/total_num_terms})
        
        repeated_terms_sum.update({f'[{st_per},{en_per}]': num_of_repeated})
        repeated_terms_average.update({f'[{st_per},{en_per}]': num_of_repeated/num_of_queries})
        repeated_terms_percentage.update({f'[{st_per},{en_per}]': num_of_repeated/total_num_terms})


        # free up memory
        del df[f'[{st_per},{en_per}]_processed']
        del df[f'[{st_per},{en_per}]_repeated']
        del df[f'[{st_per},{en_per}]_new']
        gc.collect()

     

    df_res = pd.DataFrame([queries_counter])
    df_res = pd.concat([df_res, pd.DataFrame([queries_percentage])], ignore_index=True)
    df_res = pd.concat([df_res, pd.DataFrame([new_terms_sum])], ignore_index=True)
    df_res = pd.concat([df_res, pd.DataFrame([repeated_terms_sum])], ignore_index=True)
    df_res = pd.concat([df_res, pd.DataFrame([new_terms_average])], ignore_index=True)
    df_res = pd.concat([df_res, pd.DataFrame([repeated_terms_average])], ignore_index=True)
    df_res = pd.concat([df_res, pd.DataFrame([new_terms_percentage])], ignore_index=True)
    df_res = pd.concat([df_res, pd.DataFrame([repeated_terms_percentage])], ignore_index=True)
    df_res.to_excel(save_file, index=False)
    return df_res


def process(scored_file, save_file, buckets_dir, range_step):
    '''
    scored_file: the file which contains the corpus, expansion queries with their corresponding relevance scores
    save_file: Path to save the results
                For example, if filter_type is top and the the percentage is 30, then retain queries with scrores higher than the 70 percentile
    range_step: The size of each bucket 
    '''
    
   # 1. filter based on percentage
    # 2. count number of new and copied terms of the filtered file. Set 100 for no filteration
    # 
    
    df_scored = None
    log_file = f"{scored_file}-analyse_ranges_with_step-{range_step}.log"
    logger = util.get_logger(log_file)
    logger.info(f"Reading the input file : {scored_file}")
    df_scored = pd.read_json(scored_file, lines=True, dtype={'id': str,}) 
    logger.info(f"Done reading")

    df_res= analyse_new_and_repeated_in_ranges(df=df_scored, save_file=save_file, logger=logger,
                                         buckets_dir=buckets_dir, range_step=range_step,)
    logger.info(f"Done analysis, and the results were saved to {save_file}")
    logger.info(f"results = {df_res.to_string()}")
    return df_res



def main():

    parser = argparse.ArgumentParser(description="Script to analyse relevance scores across fixed-range buckets")
    parser.add_argument("--processed_file", type=str, required=True, help="Path to the input scored and processed file")
    parser.add_argument("--save_file", type=str, required=True, help="Path to save the results")
    parser.add_argument("--range_step", type=int, default=20, required=True, help="The size of each bucket")
    parser.add_argument("--buckets_dir", type=str, default=None, required=False, help="Directory to save the resulted buckets")
    args = parser.parse_args()
    scored_file = args.processed_file
    save_file = args.save_file
    buckets_dir = args.buckets_dir
    range_step = args.range_step

    if range_step < 0 or range_step > 50:
        raise ValueError('Filter type value must be between 1 and 50')
    
    process(scored_file, save_file, buckets_dir, range_step)



if __name__ == "__main__":
    main()


