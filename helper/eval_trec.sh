trec_eval -c -l 2 -m recall.10,100,1000 $1 $2
trec_eval -c -m ndcg_cut.10 $1 $2
trec_eval -c -l 2 -M 10 -m recip_rank $1 $2