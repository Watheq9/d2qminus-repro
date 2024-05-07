# Learned Sparse Retrieval Experiments

The code and resources provided below allow partial reproduction of our LSR experiments.

Ultimately, the data files are too large to share in their current form, but we plan to release these once we have evaluated possible storage solutions.

In the meantime, the code provided here allows for the **full** end-to-end LSR reproduction.

We require pyserini for encoding uniCOIL:
```
conda activate doc2query
pip install pyserini
```

We also require the following libraries to be installed -- please follow the instructions provided by those libraries:
```
git clone https://github.com/castorini/anserini
git clone https://github.com/pisa-engine/pisa/
git clone https://github.com/osirrc/ciff
git clone https://github.com/pisa-engine/ciff pisa-ciff
git clone https://github.com/JMMackenzie/enhanced-graph-bisection/
```

The DeepImpact inference relies on the PyTerrier DeepImpact plug-in: https://github.com/terrierteam/pyterrier_deepimpact

However, we use a modified version of this plug-in that dumps the raw, inferred, json data to allow the indexing pipeline to match uniCOIL.
In the interest of anonymity, we have not yet contributed our modified version of the code to the public repository, but we provide it here.

To install the modified version, please follow these instructions:
```
git clone https://github.com/terrierteam/pyterrier_deepimpact
mv tools/di-init.py pyterrier_deepimpact/pyt_deepimpact/__init__.py
pip install pyterrier_deepimpact
```

# Experimental Process
For either uniCOIL or DeepImpact, please run `index-*.sh` followed by `query-*.sh` to replicate our experiments.
You should carefully examine the first line of the indexing script to configure your input data correctly.
