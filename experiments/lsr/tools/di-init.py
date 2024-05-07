import os
import tempfile
import pickle
import itertools
import math
from pathlib import Path
from hashlib import md5
from more_itertools import chunked
import json

import torch
import deepimpact
import deepimpact.model

from deepimpact.model import MultiBERT as DeepImpactModel
from deepimpact.utils2 import cleanD

import pyterrier as pt
from pyterrier.index import IterDictIndexer

def _load_model(path_or_url, base_model, gpu):
    from deepimpact.utils import load_checkpoint
    if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
        cache_path = Path.home() / '.pyterrier' / 'pyt_deepimpact' / md5(path_or_url.encode()).hexdigest()
        if cache_path.exists():
            print("Using cached checkpoint at %s" % cache_path)
            path_or_url = str(cache_path)
        else:
            if not cache_path.parent.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            if path_or_url.startswith('https://drive.google.com'):
                import gdown
                print("Downloading from Google drive %s" % path_or_url)
                gdown.download(path_or_url, str(cache_path), quiet=False)
            else:
                import requests, shutil
                print("Downloading %s" % path_or_url)
                try:
                    with requests.get(path_or_url, stream=True) as r, \
                         open(cache_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    path_or_url = str(cache_path)
                except:
                    if cache_path.exists():
                        cache_path.unlink()
                    raise
            path_or_url = str(cache_path)

    print("Loading checkpoint %s" % path_or_url)
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    model = DeepImpactModel.from_pretrained(base_model)
    model.to(device)
    load_checkpoint(path_or_url, model)
    model.eval()

    return model

from pyterrier.index import IterDictIndexerBase
class DeepImpactIndexer(IterDictIndexerBase):

    def __init__(self, 
                parent_indexer : IterDictIndexerBase,
                 *args,
                 batch_size=1,
                 quantization_bits=8,
                 checkpoint='https://drive.google.com/uc?id=17I2TWCB2hBSQ-E0Yt2sBEDH2z_rV0BN0',
                 base_model='bert-base-uncased',
                 gpu=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent_indexer
        self.model = _load_model(checkpoint, base_model, gpu)
        self.quantization_bits=quantization_bits
        self.batch_size=batch_size
        if not gpu:
            import deepimpact.parameters, torch
            deepimpact.model.DEVICE = deepimpact.parameters.DEVICE = torch.device("cpu")

    def index(self, doc_iter, *args, **kwargs):
        
        def _deepimpact_iter(doc_iter):

            def tok(d):
                from itertools import accumulate

                d = cleanD(d, join=False)
                content = ' '.join(d)
                tokenized_content = self.model.tokenizer.tokenize(content)

                terms = list(set([(t, d.index(t)) for t in d]))  # Quadratic!
                word_indexes = list(accumulate([-1] + tokenized_content, lambda a, b: a + int(not b.startswith('##'))))
                terms = [(t, word_indexes.index(idx)) for t, idx in terms]
                terms = [(t, idx) for (t, idx) in terms if idx < deepimpact.model.MAX_LENGTH]

                return tokenized_content, terms

            max_impact = 0.0
            with tempfile.NamedTemporaryFile() as tmp:
                from operator import itemgetter

                for batch in pt.tqdm(chunked(doc_iter, self.batch_size), desc='Computing the maximum score value and the impacts'):
                    batch = [(doc['docno'], tok(doc['text'])) for doc in batch]
                    batch = sorted(batch, key=lambda x: len(x[1][0]))
                    docnos, D = zip(*batch)
                    transformed_docs = self.model.index(D, 2 + len(D[-1][0]))
                    for docno, doc in zip(docnos, transformed_docs):
                        max_impact = max(max_impact, max(doc, key=itemgetter(1))[1])
                        pickle.dump({'docno': docno, 'text': doc}, tmp)

                # print('Max impact is', max_impact)
                scale = (1 << self.quantization_bits)/max_impact

                def quantize(transformed_doc):
                    transformed_doc = [[term] * int(math.ceil(value * scale)) for term, value in transformed_doc]
                    return ' '.join(itertools.chain.from_iterable(transformed_doc))

                encountered_docnos = set() # required to remove duplicates in cord19-like datasets :-(
                tmp.seek(0)
                out_json = open("my-deepimpact.jsonl", 'w')
                while tmp.peek(1):
                    doc = pickle.load(tmp)
                    if doc['docno'] not in encountered_docnos:
                        q_text = quantize(doc['text'])
                        x = {'docno': doc['docno'], 'text': q_text}
                        y = {'id': doc['docno'], 'contents': q_text}
                        out_json.write(json.dumps(y))
                        yield x 
                        encountered_docnos.add(doc['docno'])

        doc_iter = _deepimpact_iter(doc_iter)
        return self.parent.index(doc_iter, *args, **kwargs)
