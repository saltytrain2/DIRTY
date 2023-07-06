Building A Corpus
=================

This directory contains the files needed to generate a corpus from a set of
binaries containing debug information (i.e., compiled with `gcc -g`).
As opposed to the original implementation, this directory utilizes the open-source
[Ghidra decompiler](https://github.com/NationalSecurityAgency/ghidra), allowing reproduction of
the original paper without having to buy an IDA Pro license.
 
Prerequisites
=============

- A copy of the [Ghidra decompiler](https://github.com/NationalSecurityAgency/ghidra/releases) (tested on version 10.1.4)
- Directory of binaries containing debug information


## Binary Generation

When writing our paper, the original DIRTY team were kind enough to provide the set of binaries they compiled to train their model. They generated their dataset by feeding [GHCC](https://github.com/huzecong/ghcc) a list of github repositories to clone and compile. Similar tools can be used to generate binaries. For our paper implementation, we selected a total size of approximately 160,000 binaries to generate our dataset. A randomly-sampled binary dataset of similar magnitude should be enough.

We will release our unpreprocessed dataset when we find a suitable host platform.

Use
===

Use is fairly simple, given a directory of binaries and an existing output
directory, just run the [generate.py](generate.py) script with
Python 3:
`python3 generate.py --ghidra PATH_TO_GHIDRA -t NUM_THREADS -n [NUM_FILES|None] -b BINARIES_DIR -o OUTPUT_DIR`

This script creates a `bins/` and `types/` directory in `OUTPUT_DIR` and generates a `.jsonl` file in both directories for each binary in `BINARIES_DIR`.
The file is in the [JSON Lines](http://jsonlines.org) format, and each entry corresponds to a
function found in the binary. 
The files in the `bins/` directory contain necessary information to process the decompiled code, while the files in the `types/` directory contain all unique types in the binary.

The preprocessing script expects a file containing all filenames in your dataset. Simply run the [gen_names.py](gen_names.py) script as shown:
`python3 gen_names.py PATH_TO_DATASET`

Output Format
=============

Each line in the output is a JSON value corresponding to a function in the
binary. At the moment there are three fields for outputs in the `bins/` directory:
* `e`: Address of the function
* `b`: json representation of the decompiled function without debug information. Represents how functions decompile in the real world.
       Contains the following fields:
    * `t`: AST representation of the function. Unused.
    * `n`: Name of the function.
    * `r`: json representation of the function's return type. See `binary/ghidra_types` for more information.
    * `a`: json representation of the function's arguments/parameters.
    * `l`: json representation of the function's local variables.
    * `c`: the raw code produced by ghidra. All variables are represented as `@@id@@` for easier postprocessing.
* `c`: json representation of the decompiled function with debug information. Represents the ground truth function typing information.
       Same fields as above.

Debug Decompilation Filtering
=============================

The original DIRTY authors were hesitant to port their implementation to Ghidra because Ghidra failed more often in recovering original source types when provided debug information compared to IDA Pro.
While we don't directly improve on Ghidra's ability to 

Preprocessing
=============

After generating the dataset, the dataset is ready for preprocessing with the utility scripts in the `dirty` directory. This step maps variables
in functions decompiled without debug information to variables in functions with debug information to create testing samples for the model to train on.
