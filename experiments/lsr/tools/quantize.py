# Quantization tool
# Read a json file 
import sys
import json
import math

# Stolen from Pyserini: https://github.com/castorini/pyserini/blob/bc82ce9b679689b2c96520b1e86bdf2391753757/pyserini/index/lucene/_base.py#L183
def quantize_weights(input_file_path, output_file_path, bits = 8):
    """Takes vectors of weights in Pyserini's JSONL vector format and quantizes them.

    Parameters
    ----------
    input_file_path : str
        File path of vectors of weights in Pyserini's JSONL vector format.
    output_file_path : str
        File path to output JSONL file of quantized weight vectors.
    bits : int
        Number of bits to use to represent quantized scores.
    """

    min_weight = float('inf')
    max_weight = float('-inf')

    input_file = open(input_file_path, 'r')

    # vectors are read line by line to avoid running out of memory
    for line in input_file:
        doc = json.loads(line)
        for weight in doc['vector'].values():
            if weight > max_weight:
                max_weight = weight
            if weight < min_weight:
                min_weight = weight
    input_file.seek(0)

    output_file = open(output_file_path, 'w')

    smallest_impact = 1
    for line in input_file:
        doc = json.loads(line)
        for element in doc['vector']:
            doc['vector'][element] = math.floor((2 ** bits - smallest_impact) * (doc['vector'][element] - min_weight) / (max_weight - min_weight)) + smallest_impact
        output_file.write(json.dumps(doc) + "\n")

    input_file.close()
    output_file.close()


dump_file_path = sys.argv[1]
quantized_file_path = sys.argv[2]
quantize_weights(dump_file_path, quantized_file_path)

