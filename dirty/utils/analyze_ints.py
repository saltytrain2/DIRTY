import argparse
import json
from collections import defaultdict

import _jsonnet
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils.dataset import Dataset   

UNSIGNED_INT_TYPES = {"uint", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t", "ulong", "uchar",
                      "uint32", "uint16", "uint8", "uint64", "UINT32", "UINT16", "UINT8", "UINT64", 
                      "u_int", "__u32", "__u16", "__u8", "__u64", "u32", "u16", "u8", "u64", "U32",
                      "U16", "U8", "U64"}

SIGNED_INT_TYPES = {"int", "int8_t", "int16_t", "int32_t", "int64_t", "long", "long long", "short", "char",
                    "int32", "int16", "int8", "int64", "INT32", "INT16", "INT8", "INT64", "INTEGER", "s32",
                    "s16", "s8", "s64", "S32", "S16", "S8", "S64", "__s32", "__s16", "__s8", "__s64"}

def add_options(parser):
    parser.add_argument(
        "--pred-file", type=str, required=True, help="Saved predictions on a dataset"
    )
    parser.add_argument("--config-file", type=str, required=True)

def load_data(config_file):
    config = json.loads(_jsonnet.evaluate_file(config_file))["data"]
    config["max_num_var"] = 1 << 30
    dataset = Dataset(config["test_file"], config)
    return dataset

def no_mask(test_metas):
    return np.array([True for _ in test_metas])

def body_not_in_train_mask(test_metas):
    return np.array([
        not test_meta["function_body_in_train"] for test_meta in test_metas
    ])

def get_int_acc(preds, results, is_unsigned, functions=[], is_width_sensitive = False):
    if is_unsigned:
        int_filter = UNSIGNED_INT_TYPES
        opposite_filter = SIGNED_INT_TYPES
    else:
        int_filter = SIGNED_INT_TYPES
        opposite_filter = UNSIGNED_INT_TYPES
    
    int_mask = np.array([ref in int_filter for ref in results])
    masked_preds = preds[int_mask]
    correct_sign = np.array([pred in int_filter for pred in masked_preds])
    incorrect_sign = np.array([pred in opposite_filter for pred in masked_preds])
    correct_sign_acc = correct_sign.mean()
    incorrect_sign_percentage = incorrect_sign.mean()

    
    if is_unsigned:
        print(f"unsigned accuracy: {correct_sign_acc}")
        print(f"incorrect type as signed: {incorrect_sign_percentage}")
    else:
        print(f"signed accuracy: {correct_sign_acc}")
        print(f"incorrect type as unsigned: {incorrect_sign_percentage}")

    if is_unsigned:
        mislabeled_unsigned_functions = functions[int_mask]
        mislabeled_unsigned_functions = mislabeled_unsigned_functions[incorrect_sign]
        with open("unsigned_mislabel_projects.txt", "w") as f:
            for func in mislabeled_unsigned_functions:
                f.write(func + "\n")
        pass

    pass

def evaluate(dataset, results):
    ccode = defaultdict(dict)
    pred_names, ref_names, pred_types, ref_types, src_types = [], [], [], [], []
    test_meta_types, test_meta_names = [], []
    function_name = []

    for example in tqdm(dataset):
        for src_name, src_type, tgt_name, tgt_type in zip(
            example.src_var_names, example.src_var_types_str, example.tgt_var_names, example.tgt_var_types_str
        ):
            pred_type, _ = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )
         
         
            pred_types.append(pred_type)
            ref_types.append(tgt_type)
            function_name.append(example.binary + " " + example.name)
            src_types.append(src_type)
            test_meta = example.test_meta.copy()
            test_meta_types.append(test_meta)

            if src_name != tgt_name and tgt_name != "@@@@":
                # only report need_rename
                _, pred_name = (
                    results.get(example.binary, {})
                    .get(example.name, {})
                    .get(src_name[2:-2], ("", ""))
                )
                pred_names.append(pred_name)
                ref_names.append(tgt_name[2:-2])
                test_meta_names.append(test_meta)

    pred_types = np.array(pred_types, dtype=object)
    ref_types = np.array(ref_types, dtype=object)
    pred_names = np.array(pred_names, dtype=object)
    ref_names = np.array(ref_names, dtype=object)
    src_types = np.array(src_types, dtype=str)
    function_name = np.array(function_name, dtype=str)

    mask = body_not_in_train_mask(test_meta_types)
    pred_types = pred_types[mask]
    ref_types = ref_types[mask]
    function_name = function_name[mask]
    src_types = src_types[mask]
    mask = body_not_in_train_mask(test_meta_names)
    pred_names = pred_names[mask]
    ref_names = ref_names[mask]

    print(f"% src unsigned: {np.array([src_type in UNSIGNED_INT_TYPES for src_type in src_types]).mean()}") 
    print(f"% src signed: {np.array([src_type in SIGNED_INT_TYPES for src_type in src_types]).mean()}")
    get_int_acc(pred_types, ref_types, True, function_name)
    get_int_acc(pred_types, ref_types, False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    results = json.load(open(args.pred_file))
    dataset = load_data(args.config_file)

    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    evaluate(dataset, results)