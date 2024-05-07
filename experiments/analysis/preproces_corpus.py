import os, sys
sys.path.insert(0, os.getcwd())
import pandas as pd
import helper.util as util
import helper.preprocessing as preprocessing
import argparse

def get_split_word():
    return "mx12e34m"


def join_queries_with_word(queries, join_word=""):
    if join_word == "":
        join_word = get_split_word()
    # join word: is a non-meaning word to make the separation possible if needed
    return f" {join_word} ".join(queries)


def preprocess_queries(queries):

    queries_str = join_queries_with_word(queries, join_word=get_split_word())
    # preprocessing on string is much faster than processing a list of strings (queries)
    processed_query = preprocessing.preprocess(queries_str)
    processed_queries = processed_query.split(get_split_word())
    return processed_queries
 

def get_unique_terms(text):
    processed_text = preprocessing.preprocess(text, stop_words='terrier')
    terms = processed_text.split(' ')
    unique_terms = list(set(terms))
    # unique_terms = ' '.join(unique_terms)
    return unique_terms


def preprocess_corpus(input_file, save_file, logger):

    df = pd.read_json(input_file, lines=True,)
    logger.info(f"Loaded the document with expansion queries file from this path {input_file}")

    cnt = 0
    for row in df.itertuples():
        doc_unique_terms = get_unique_terms(row.text)
        queries_processed = preprocess_queries(row.predicted_queries)
        new_json_obj = {'id': str(row.id), 
                        "doc_unique_processed_terms": doc_unique_terms,
                        "queries_processed": queries_processed,
                        "querygen_score": row.querygen_score,
                        }
        
        util.write_new_line_to_jsonl(file_path=save_file, new_line=new_json_obj)

        cnt += 1
        if cnt % 10000 == 0:
            logger.info(f"Processed {cnt} so far out of {len(df)}")

    logger.info(f"Done processing the whole corpus")



def main():

    parser = argparse.ArgumentParser(description="Preprocessing collection with expansion queries")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data file that contains four columns (id, text, predicted_queries, querygen_score)")
    parser.add_argument("--output", type=str, required=True, help="Path to the save the output file")
    args = parser.parse_args()
    input = args.input
    output = args.output
    log_file = f"{output}.log"
    
    logger = util.get_logger(log_file)
    logger.info(f"The input file is read from : {input}")
    logger.info(f"The log file is saved into : {log_file}")
    logger.info(f"The output file is saved into : {output}")

    preprocess_corpus(input_file=input, save_file=output, logger=logger)


if __name__ == "__main__":
    main()

