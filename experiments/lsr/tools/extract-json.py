import numpy as np
import os
import pandas as pd
import sys

"""
Take a JSON file with predicted queries, normalize them to find only the unique
terms that were not in the original document, and then append them to the end
of the given passage - this follows the uniCOIL training pipeline.
"""

# DeepImpact repo: https://github.com/DI4IR/SIGIR2021/blob/a6b1ee4efaba7d0de75501f2f05a4b9353cdb673/src/utils2.py
printable = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
printableX = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-. ')
printable3X = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- ')
printableD = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
printable3D = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- ')
def cleanD(s, join=True):
    s = [(x.lower() if x in printable3X else ' ') for x in s]
    s = [(x if x in printableX else ' ' + x + ' ') for x in s]
    s = ''.join(s).split()
    s = [(w if '.' not in w else (' . ' if len(max(w.split('.'), key=len)) > 1 else '').join(w.split('.'))) for w in s]
    s = ' '.join(s).split()
    s = [(w if '-' not in w else w.replace('-', '') + ' ( ' + ' '.join(w.split('-')) + ' ) ') for w in s]
    s = ' '.join(s).split()
    # s = [w for w in s if w not in STOPLIST]
    return ' '.join(s) if join else s

def join_text_and_queries(text, queries, sep):
    terms = set(cleanD(text, False))
    inject = set(cleanD(queries, False))
    inject = inject.difference(terms)
    septext = ' '
    if sep:
        septext = ' [SEP] '
    return text + septext + str(" ".join(inject))

use_sep = False
if len(sys.argv) > 3 and sys.argv[3] == "--sep":
    print ("Using a [SEP] token between document and expansion terms.")
    use_sep = True
else:
    print ("Not using a [SEP] token between document and expansion terms.")

print("Reading...")
all_file = str(sys.argv[1])
df_filt = pd.read_json(all_file, orient='records', lines=True)
print("Read.")

print("Injecting...")
df_idx = pd.DataFrame()
df_idx['docno'] = df_filt['docno'].apply(lambda x: str(x))
df_idx['text'] = df_filt.apply(lambda row: join_text_and_queries(row['text'], 
                                            row['predicted_queries'], use_sep), axis=1)
print("Injected.")
del df_filt

print("Writing...")
with open(sys.argv[2], 'w') as file:
    df_idx.to_json(file, orient="records", lines=True, force_ascii=False)
print("Done!")
