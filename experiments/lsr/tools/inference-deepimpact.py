import sys
from pyterrier_pisa import PisaIndex
pt.init() 
from pyt_deepimpact import DeepImpactIndexer # Must use our local version!

# the input dataframe should have 'docno' as the id column of the documents
def build_pyterrier_index(df, index_path):
    parent = pt.index.IterDictIndexer(index_path, verbose=True, overwrite=True)
    parent.setProperty("termpipelines", "")
    # perform indexing
    indexer = DeepImpactIndexer(parent, batch_size=32)
    index_ref = indexer.index(df.to_dict(orient="records"), meta=['docno'])
    return indexer

df_filt = pd.read_json(sys.argv[1], orient='records', lines=True)
df_idx = pd.DataFrame()
df_idx['docno'] = df_filt['docno']
df_idx['text'] = df_filt.apply(lambda row: join_text_and_queries(row['text'], 
                                            row['predicted_queries']), axis=1)

pisa_index = build_pyterrier_index(df=df_idx, index_path="/dev/null/")
