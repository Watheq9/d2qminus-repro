# Run Files

We provide raw run files for each of our experimental approaches.
All run files correspond to a system on either `dev` or `dl19` or `dl20` respectively.

Please download and extract the runs file from the following URL: 

https://www.dropbox.com/scl/fi/0i2v15qqc1yu9yo2ytgcj/doc2query-repro-runs.zip?rlkey=9meum1gs08xkhlg91jo5qc2ag&dl=0

## Main Experiment

See `main/` directory:

- `bm25` is the tuned `bm25` baseline run;
- `dtq` is the tuned `docTTTTTquery` run with N=80;
- `baseline` is the `doc2query--` run provided by the authors;
- `minusti` is our `doc2query--` run using the index provided by the authors;
- `minusdef` is our reproduced `doc2query--` run using our own index with default parameters;
- `minustune` is our reproduced `doc2query--` run using our own index with tuned parameters;
- `minuslocal` is our doc2query--LF local filtering system;
- `rl` is our `RLGen` system;

## Re-ranking

See the `rerank/` directory. All runs are re-ranked:

- `bm25` is as above;
- `dtq` is as above;
- `dtqminus` is the same as the `minustune` run above;
- `dtqminuslf` is the same as the `minuslocal` run above.


## Learned Sparse

See `lsr/` directory:

- `ww-*` correspond to the baseline N=40 runs of both DeepImpact and UniCOIL;
- `-orig-*` correspond to the reproduced N=80 runs;
- `*-70-high*` correspond to the filtered LSR systems.
