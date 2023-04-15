import argparse
import json
from collections import defaultdict

import _jsonnet
import numpy as np
from tqdm import tqdm

from utils.dataset import Dataset

def add_options(parser):
    parser.add_argument(
        "--ghidra-file", type=str, required=True, help="Location of ghidra dataset"
    )
    parser.add_argument(
        "--ida-file", type=str, required=True, help="Location of ida dataset"
    )

def analyze_overlap(ghidra_dataset, ida_dataset):
    ida_binaries = set()
    ida_functions = set()

    for example in tqdm(ida_dataset):
        ida_binaries.add(example.binary)
        ida_functions.add(example.binary + example.name)
    
    ghidra_binaries = set()
    ghidra_functions = set()
    
    for example in tqdm(ghidra_dataset):
        ghidra_binaries.add(example.binary)
        ghidra_functions.add(example.binary + example.name)
    
    overlap_binaries = set.intersection(ida_binaries, ghidra_binaries)
    overlap_functions = set.intersection(ida_functions, ghidra_functions)

    print(f"total number of ida binaries: {len(ida_binaries)}")
    print(f"total number of ida functions: {len(ida_functions)}")
    print(f"total number of ghidra binaries: {len(ghidra_binaries)}")
    print(f"total number of ghidra functions: {len(ghidra_functions)}")
    print(f"total number of overlapping binaries: {len(overlap_binaries)}")
    print(f"total number of overlap functions: {len(overlap_functions)}")

    pass

def main():
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    ida_dataset = Dataset(args.ida_file)
    ghidra_dataset = Dataset(args.ghidra_file)

    from torch.utils.data import DataLoader

    ida_dataset = DataLoader(ida_dataset, num_workers=8, batch_size=None)
    ghidra_dataset = DataLoader(ghidra_dataset, num_workers=8, batch_size=None)

    analyze_overlap(ghidra_dataset, ida_dataset)
    pass

if __name__ == "__main__":
    main()
