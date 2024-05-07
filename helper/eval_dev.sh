trec_eval -c -M 10 -m recip_rank $1 $2
trec_eval -c -m recall.10,100,1000 $1 $2
trec_eval -c -m ndcg_cut.10 $1 $2
