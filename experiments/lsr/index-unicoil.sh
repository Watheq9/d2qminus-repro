# As in the main experiments, you need to prepare a .jsonl file that contains three columns ['id', 'text', 'predicted_queries']. We will call this input.jsonl
# Note that this program assumes you have already filtered the `predicted_queries` column to leave only the top-70%, but you are free to supply whatever predicted queries you wish.

echo "Convert JSON for inference..."
python3 tools/extract-json.py input.json unicoil-prepared.jsonl

echo "Encoding uniCOIL with top 70%"
CUDA_VISIBLE_DEVICES=1 python3.10 -m pyserini.encode input --corpus unicoil-prepared.jsonl --fields text --docid-field docno output --embeddings unicoil-top70 encoder --fp16 --encoder castorini/unicoil-msmarco-passage --batch 32

echo "Done: Embedding json file is located in unicoil-top70/embeddings.jsonl"

echo "Quantizing embeddings..."
python3.10 tools/quantize.py unicoil-top70/embeddings.jsonl quantized.jsonl

mkdir target
mv quantized.jsonl target

echo "Indexing the embeddings..."

echo "Using anserini to read the quantized json format..."
./anserini/target/appassembler/bin/IndexCollection -generator DefaultLuceneDocumentGenerator -collection JsonVectorCollection -input target -index anserini-top70 -impact -optimize -pretokenized

echo "Convert the index to CIFF"
ciff/target/appassembler/bin/ExportAnseriniLuceneIndex -index anserini-top70/ -output unicoil-top70.ciff

echo "Reorder the index"
enhanced-graph-bisection/target/release/create-rgb -i unicoil-top70.ciff -o bp-top70.ciff -m 256 --loggap

echo "Replace CIFF with reordered ciff..."
mv bp-top70.ciff unicoil-top70.ciff

echo "Convert to PISA..."
mkdir pisa-index
./pisa-ciff/target/release/ciff2pisa --ciff-file unicoil-top70.ciff --output pisa-index/unicoil-top70
   
echo "Building PISA index with block_simdbp compression..."
./pisa/build/bin/compress_inverted_index --encoding block_simdbp --collection pisa-index/unicoil-top70 --output pisa-index/unicoil-top70.block_simdbp.idx

echo "Building WAND data..."

./pisa/build/bin/create_wand_data --collection pisa-index/unicoil-top70 \
                                  --output pisa-index/unicoil-top70.bmw \
                                  --block-size 40 \
                                  --scorer quantized 

echo "Building the term lexicon..."
./pisa/build/bin/lexicon build pisa-index/unicoil-$f-$t.terms pisa-index/unicoil-$f-$t.termlex

echo "Building the document identifier map..."
./pisa/build/bin/lexicon build pisa-index/unicoil-$f-$t.documents pisa-index/unicoil-$f-$t.docmap
echo "Done."

