import os
import pandas as pd
from pyterrier.measures import *
import pyterrier as pt
import numpy as np
import traceback
import gc
from pyterrier_pisa import PisaIndex
import re
import subprocess 
import configure as cf
if not pt.started():
    pt.init(version='5.7', helper_version='0.0.7')


MS_MARCOV1 = "irds:msmarco-passage/"
TREC_2019 = 'irds:msmarco-passage/trec-dl-2019/judged'
TREC_2020 = 'irds:msmarco-passage/trec-dl-2020/judged'
DEV1 = 'irds:msmarco-passage/dev/small'
DEV2 = 'irds:msmarco-passage/dev/2'
TREC19_QRELS = cf.TREC19_QRELS
TREC20_QRELS = cf.TREC20_QRELS
DEV1_QRELS = cf.DEV1_QRELS
DEV2_QRELS = cf.DEV2_QRELS
eval_dev = cf.eval_dev
eval_trec  = cf.eval_trec 


def get_test_sets_and_measures():
    sets = [ 'irds:msmarco-passage/dev/small', 'irds:msmarco-passage/dev/2',
        'irds:msmarco-passage/trec-dl-2019/judged', 'irds:msmarco-passage/trec-dl-2020/judged',]
    measures = [RR@10, RR@10, nDCG@10, nDCG@10,]
    return sets,  measures
        
def save_file(df, save_file):
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    
    df.to_csv(save_file, index=False)


def get_eval_metrics():
    return [RR@10, nDCG@10, R@10, R@100, R@1000]


def evaluate_run(df_run, test_set='irds:msmarco-passage/trec-dl-2019/judged', name='BM25'):
    dataset = pt.get_dataset(test_set)
    measures = get_eval_metrics()
    df_res = pt.Experiment(
        [df_run],
        dataset.get_topics(),
        dataset.get_qrels(),
        measures,
        round=3,
        names=[name]
        )
    new_row = {'name': "BM25", 
                'test': test_set, 
                "RR@10": df_res["RR@10"].values[0],
                "nDCG@10": df_res["nDCG@10"].values[0],
                "R@10": df_res["R@10"].values[0],
                "R@100": df_res["R@100"].values[0],
                "R@1000": df_res["R@1000"].values[0],
                }
    return new_row

def evalate_by_trec(script_path, qrels, run):
    try:
        result = subprocess.check_output(['bash', script_path, qrels, run,], universal_newlines=True)
        # print("Script output:\n", result)
    except subprocess.CalledProcessError as e:
        print("Error running the script:", e)

    output_lines = result.split('\n')
    eval_scores = {}

    for line in output_lines:
        line = re.sub(r"\t", " ", line)  # remove tabs
        line = re.sub(r"\s+", " ", line)  # remove extra white space
        # text = re.sub(r"\n", " ", text)  # remove line jump
        vals = line.split(' all ')
        if len(vals) == 2:
            eval_scores.update({vals[0]: vals[1]})
    
    return eval_scores


def get_evaluation_row(run, test_set, name="BM25", b=None, k=None):
    
    if test_set == TREC_2019:
        result = evalate_by_trec(script_path=eval_trec, qrels=TREC19_QRELS, run=run)
    elif test_set == TREC_2020:
        result = evalate_by_trec(script_path=eval_trec, qrels=TREC20_QRELS, run=run)
    elif test_set == DEV1:
        result = evalate_by_trec(script_path=eval_dev, qrels=DEV1_QRELS, run=run)
    elif test_set == DEV2:
        result = evalate_by_trec(script_path=eval_dev, qrels=DEV2_QRELS, run=run)

    result.update({'name': name, 
                    'test': test_set, 
                    "run": run})
    if b is not None:
        result.update({'b': b,})
    if b is not None:
        result.update({'k': k,})    

    return result



def evaluate_model(test_set, measure, model, run_save_dir=None, run_name=None, 
                   query_column=None, qrels=None, queries=None):                
    measures = [RR@10, nDCG@10, R@10, R@100, R@1000, P@10, P@100]
    if measure not in measures:
        measures.append(measure)

    if qrels is None:
        dataset = pt.get_dataset(test_set)
        qrels = dataset.get_qrels()
    if queries is None:
        dataset = pt.get_dataset(test_set)
        queries = dataset.get_topics(query_column)
         

    if run_save_dir is not None and not os.path.exists(run_save_dir):
        os.makedirs(run_save_dir)

    if run_save_dir is not None:

        df_curr = pt.Experiment(
            [model],
            queries,
            qrels,
            measures,
            round=3,
            save_dir=run_save_dir,
            save_mode='overwrite',
            names=[run_name])
    else:
        df_curr = pt.Experiment(
            [model],
            queries,
            qrels,
            measures,
            round=3,)

    return df_curr


def form_BM25_evaluation_row(df_res, test_set='', name='BM25', b=None, k=None):
    return {'name': name, 
            'test': test_set, 
            'b': b,
            'k': k,
            "RR@10": df_res[str("RR@10")].values[0],
            "nDCG@10": df_res[str("nDCG@10")].values[0],
            "R@10": df_res[str("R@10")].values[0],
            "R@100": df_res[str("R@100")].values[0],
            "R@1000": df_res[str("R@1000")].values[0],
            }


def tune_b_and_k1(index_path, dev, measure, retr_name='BM25', runs_dir=None, 
                  save_path="", logger=None):
    
    ks = np.round(np.arange(0.5, 2.55, 0.25), decimals=2).tolist()
    bs = np.round(np.arange(0.0, 1.05, 0.10), decimals=2).tolist()
    df_res = pd.DataFrame()
    best_k = -1
    best_b = -1
    best_score = -1
    best_recall10 = -1
    best_recall100 = -1
    print("Start tuning b & K")
    for b in bs:
        pisa_index = PisaIndex(index_path, stops='none')
        for k in ks:
            print(f"Processing b = {b}, k = {k}")
            bm25 = pisa_index.bm25(b=b, k1=k)
            if runs_dir is not None:
                dev_name = dev.replace("irds:msmarco-passage/", "")
                dev_name = dev_name.replace("/", "-")
                run_name = f"BM25_k={k}-b={b}-on-{dev_name}"
                df_c = evaluate_model(test_set=dev, measure=measure, model=bm25,
                                run_save_dir=runs_dir, run_name=run_name)
            else:
                df_c = evaluate_model(test_set=dev, measure=measure, model=bm25)
                
            score = df_c[str(measure)].values[0]
            recall_10 = df_c["R@10"].values[0]
            recall_100 = df_c["R@100"].values[0]
            new_row = form_BM25_evaluation_row(df_c, dev, retr_name, b, k)
            df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
            
            if score > best_score:
                best_b = b
                best_k = k
                best_score = score
                best_recall10 = recall_10
                best_recall100 = recall_100
            elif score == best_score:
                if recall_10 > best_recall10:
                    best_b = b
                    best_k = k
                    best_score = score
                    best_recall10 = recall_10
                    best_recall100 = recall_100
                elif recall_10 == best_recall10:
                    if recall_100 > best_recall100:
                        best_b = b
                        best_k = k
                        best_score = score
                        best_recall10 = recall_10
                        best_recall100 = recall_100
            print(f"Done processing b = {b}, k = {k}")
    if save_path != "":
        save_file(df_res, save_path)
        
    return best_b, best_k, df_res, best_score


def unzip_file(zipped_file, unzipped_file):
    try:
        command = f"gunzip -c {zipped_file} > {unzipped_file}"
        os.system(command)
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False


def get_run_file(runs_dir, run_name):
    zipped_run = os.path.join(runs_dir, f"{run_name}.res.gz")
    unzipped_run = os.path.join(runs_dir, f"{run_name}.run")
    unzip_file(zipped_file=zipped_run, unzipped_file=unzipped_run)
    return unzipped_run


def get_best_b_and_k(tuning_file):
    df = pd.read_csv(tuning_file)
    # Sort the DataFrame by the target measures
    sorted_df = df.sort_values(by=['RR@10', 'R@10', 'R@100'], ascending=[False, False, False])
    result = sorted_df.head(1)
    best_k = result['k'].values[0]
    best_b = result['b'].values[0]
    return best_b, best_k

def test_BM25_and_save_runs(index_path, b, k, runs_dir, run_initial_name, eval_save_file,
                            dataset=None, query_column=None, qrels=None, queries=None):
    
    model_name = "BM25"
    if isinstance(index_path, PisaIndex):
        pisa_index = index_path
    else:
        pisa_index = PisaIndex(index_path, stops='none')
    
    bm25 = pisa_index.bm25(b=b, k1=k)
    df_res = pd.DataFrame()
    if dataset is None: # the default evaluation is on ms marco v1
        dataset = MS_MARCOV1
        tests, measures = get_test_sets_and_measures()
    else:
        tests = [dataset]
        measures = [RR@10]

    for test, measure in zip(tests, measures):
        test_name = dataset
        if dataset == MS_MARCOV1:
            test_name = test.replace(dataset, "") # just for brevity
        test_name = test_name.replace("/", "-")
        test_name = test_name.replace(":", "-")
        run_name = f"{run_initial_name}-on-{test_name}"
        df_c = evaluate_model(test_set=test, measure=measure, model=bm25, 
                            run_save_dir=runs_dir, run_name=run_name, 
                            query_column=query_column, qrels=qrels, queries=queries)

        run_file = get_run_file(runs_dir, run_name)
        if dataset == MS_MARCOV1: # special handling for evaluating recall on trec test collections
            new_row = get_evaluation_row(run_file, test, name=model_name, b=b, k=k)
            df_res = pd.concat([df_res, pd.DataFrame([new_row])], ignore_index=True)
        else: 
            df_c['run'] = [run_file] * len(df_c) # save the run path with the results
            df_res = pd.concat([df_res, df_c], ignore_index=True)
    
    save_file(df_res, eval_save_file)
    return df_res