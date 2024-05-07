# As in the main experiments, you need to prepare a .jsonl file that contains three columns ['id', 'text', 'predicted_queries']. We will call this input.jsonl
# Note that this program assumes you have already filtered the `predicted_queries` column to leave only the top-70%, but you are free to supply whatever predicted queries you wish.

echo "Convert JSON for inference..."
python3 tools/extract-json.py input.json deepimpact-prepared.jsonl --sep # Note that DeepImpact expects a [SEP] token before expansion terms are added.

echo "Encoding DeepImpact with top 70%"
CUDA_VISIBLE_DEVICES=1 python3 tools/inference-deepimpact.py deepimpact-prepared.jsonl

echo "Done: Embedding json file is located in my-deepimpact.jsonl"

echo "Quantizing embeddings..."
python3 tools/quantize.py my-deepimpact.jsonl quantized.jsonl

mkdir target
mv quantized.jsonl target

echo "Indexing the embeddings..."

echo "Using anserini to read the quantized json format..."
./anserini/target/appassembler/bin/IndexCollection -generator DefaultLuceneDocumentGenerator -collection JsonVectorCollection -input target -index anserini-top70 -impact -optimize -pretokenized

echo "Convert the index to CIFF"
ciff/target/appassembler/bin/ExportAnseriniLuceneIndex -index anserini-top70/ -output deepimpact-top70.ciff

echo "Reorder the index"
enhanced-graph-bisection/target/release/create-rgb -i deepimpact-top70.ciff -o bp-top70.ciff -m 256 --loggap

echo "Replace CIFF with reordered ciff..."
mv bp-top70.ciff deepimpact-top70.ciff

echo "Convert to PISA..."
mkdir pisa-index
./pisa-ciff/target/release/ciff2pisa --ciff-file deepimpact-top70.ciff --output pisa-index/deepimpact-top70
   
echo "Building PISA index with block_simdbp compression..."
./pisa/build/bin/compress_inverted_index --encoding block_simdbp --collection pisa-index/deepimpact-top70 --output pisa-index/deepimpact-top70.block_simdbp.idx

echo "Building WAND data..."

./pisa/build/bin/create_wand_data --collection pisa-index/deepimpact-top70 \
                                  --output pisa-index/deepimpact-top70.bmw \
                                  --block-size 40 \
                                  --scorer quantized 

echo "Building the term lexicon..."
./pisa/build/bin/lexicon build pisa-index/deepimpact-$f-$t.terms pisa-index/deepimpact-$f-$t.termlex

echo "Building the document identifier map..."
./pisa/build/bin/lexicon build pisa-index/deepimpact-$f-$t.documents pisa-index/deepimpact-$f-$t.docmap
echo "Done."

