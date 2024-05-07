mkdir runs

collection=unicoil-top70
for topics in dl19 dl20 dev; do
      
  echo "Run: $collection -> $f $t, on $topics"
  ./pisa/build/bin/evaluate_queries --encoding block_simdbp \
                                    --documents pisa-index/$collection.docmap \
                                    --index pisa-index/$collection".block_simdbp.idx" \
                                    --wand pisa-index/$collection.fixed-40.bmw \
                                    --terms pisa-index/$collection.termlex \
                                    --algorithm maxscore \
                                    -k 1000 \
                                    --scorer quantized \
                                    --weighted \
                                    --tokenizer whitespace \
                                    --queries queries/unicoil/$topics.pisa \
                                    --run "$collection" > runs/$collection"."$topics".trec"

done

