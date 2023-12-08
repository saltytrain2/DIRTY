# This is a script to generate the pred-mt-ref.json file used by prepare_vis.py
# script for the DIRTY explorer web interface.

import argparse

from collections import defaultdict
from .evaluate import load_data

import json
import tqdm

def add_options(parser):
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="pred_mt_ref.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    dataset = load_data(args.config_file)

    d = defaultdict(dict)

    for e in tqdm.tqdm(dataset):
        binary = e.binary
        func = e.name
        body_in_train = e.test_meta['function_body_in_train']
        d[binary][func] = {}

        for srcname, tgtname, tgttyp in zip(e.src_var_names, e.tgt_var_names, e.tgt_var_types_str):
            d[binary][func][srcname[2:-2]] = (tgtname, tgttyp, body_in_train)

    open(args.output_file, "w").write(json.dumps(d, indent=2))

