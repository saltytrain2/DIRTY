import argparse
import json
from collections import defaultdict

import _jsonnet
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils.dataset import Dataset

class ExampleFailure:
    def __init__(self, code_tokens):
        self.code = unlex(code_tokens)
        self.failures = []

    def add_failure(self, src_name, pred, tgt):
        self.failures.append((src_name, pred, tgt))
        
    def get_num_failures(self):
        return len(self.failures)

def unlex(tokens):
    func = []
    cur_indent = 0
    for token in tokens:
        if token == "}":
            cur_indent -= 1
            func[-1] = "\t" * cur_indent
        func.append(token)
        func.append(" ")
        if token == "{" or token == ";" or token == "}":
            func.append("\n")
            if token == "{":
                cur_indent += 1
            func.append("\t" * cur_indent)

    return "".join(func)
        

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

def body_in_train_mask(test_metas):
    return np.array([
        test_meta["function_body_in_train"] for test_meta in test_metas
    ])

def body_not_in_train_mask(test_metas):
    return np.array([
        not test_meta["function_body_in_train"] for test_meta in test_metas
    ])

def is_unk_mask(test_metas):
    return np.array(
        [
            test_meta["is_unk"] for test_meta in test_metas
        ]
    )

def is_empty_mask(test_metas):
    return np.array(
        [
            test_meta["is_empty"] for test_meta in test_metas
        ]
    )

def shrink_errors(errors, num=5):
    for key in errors:
        total_errors = sum(errors[key].values())
        # up to num frequent guesses are saved
        frequent_predictions = sorted(errors[key].items(), key=lambda kv: kv[1], reverse=True)[:num]

        errors[key] = {k:(v/total_errors) for k, v in frequent_predictions}
        errors[key]["total"] = total_errors

def dump_freq(preds, results, filename, test_meta, mask_fun):
    def to_dict(ddict):
        ddict = dict(ddict)

        for key in ddict:
            ddict[key] = dict(ddict[key])

    errors = defaultdict(lambda: defaultdict(int))
    
    mask = mask_fun(test_meta)
    masked_preds = preds[mask]
    masked_results = results[mask]
    for pred, tgt in zip(masked_preds, masked_results):
        errors[tgt][pred] += 1
    
    to_dict(errors)
    shrink_errors(errors)

    with open(filename, "w") as f:
        json.dump(errors, f)

    return errors

def dump_errors(preds, results, filename, test_meta, ccodes, src_names, functions, mask_fun):
    errors = defaultdict(list)

    mask = mask_fun(test_meta) & (preds != results)
    masked_preds = preds[mask]
    masked_results = results[mask]
    masked_functions = functions[mask]
    masked_src_names = src_names[mask]
    print(len(masked_preds))

    for pred, tgt, function, src_name in zip(masked_preds, masked_results, masked_functions, masked_src_names):
        errors["ccode"].append(ccodes[function[:64]][function[64:]])
        errors["src"].append(src_name)
        errors["pred"].append(pred)
        errors["tgt"].append(tgt)
    
    errors = dict(errors)

    df = pd.DataFrame(data=errors)
    df.to_excel(filename, sheet_name="Sheet 1")

def to_excel(predictions, filename):
    df = pd.DataFrame()

    for key in predictions:
        prediction = predictions[key]
        cur_pred_df = {"tgt":[key], "total": [prediction["total"]]}
        del prediction["total"]
        for i, pred in enumerate(prediction):
            cur_pred_df[f"pred_{i + 1}"] = [pred]
            cur_pred_df[f"rate_{i + 1}"] = [prediction[pred]]
        df = pd.concat([df, pd.DataFrame(data=cur_pred_df)])
    
    df.to_excel(filename, sheet_name="Sheet 1")

def dump_failures(example_failures, max_len, filename):
    df = pd.DataFrame()

    for failure in example_failures:
        cur_df = {"code": [failure.code]}
        i = 1
        for src, pred, tgt in failure.failures:
            cur_df[f"src_{i}"] = [src]
            cur_df[f"pred_{i}"] = [pred]
            cur_df[f"tgt_{i}"] = [tgt]
            i += 1

        while i <= max_len:
            cur_df[f"src_{i}"] =[]
            cur_df[f"pred_{i}"] = []
            cur_df[f"tgt_{i}"] =[]
            i += 1

        df = pd.concat([df, pd.DataFrame(data=cur_df)])

    df.to_excel(filename, sheet_name="Sheet 1")
    pass
    
def evaluate(dataset, results):
    example_failures = []
    functions = []
    ccode = defaultdict(dict)
    pred_names, ref_names, pred_types, ref_types, src_names_types, src_names_names = [], [], [], [], [], []
    test_meta_types, test_meta_names = [], []
    max_len = 0

    for example in tqdm(dataset):
        failure = ExampleFailure(example.code_tokens)
        for src_name, tgt_name, tgt_type in zip(
            example.src_var_names, example.tgt_var_names, example.tgt_var_types_str
        ):
            pred_type, _ = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )
            
            # if str(pred_type) == "":
            #     print(".", end="")
            #     if (len(example.code_tokens) > 512):
            #         print(example.binary)
            #         print(unlex(example.code_tokens))
            #         exit(0)

            functions.append(example.binary + example.name)
            if example.name not in ccode[example.binary]:
                ccode[example.binary][example.name] = unlex(example.code_tokens)
            pred_types.append(pred_type)
            ref_types.append(tgt_type)
            src_names_types.append(src_name[2:-2])
            test_meta = example.test_meta.copy()
            test_meta["is_empty"] = str(pred_type) == ""
            test_meta["is_unk"] = str(pred_type) == "<unk>"
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
                src_names_names.append(src_name[2:-2])
                test_meta_names.append(test_meta)

    pred_types = np.array(pred_types, dtype=object)
    ref_types = np.array(ref_types, dtype=object)
    pred_names = np.array(pred_names, dtype=object)
    ref_names = np.array(ref_names, dtype=object)
    src_names_types = np.array(src_names_types, dtype=str)
    src_names_names = np.array(src_names_names, dtype=str)
    functions = np.array(functions, dtype=str)

    # type_not_in_train = dump_freq(pred_types, ref_types, "pred_types_common.json", test_meta_types, body_not_in_train_mask)
    # name_not_in_train = dump_freq(pred_names, ref_names, "pred_names_common.json", test_meta_names, body_not_in_train_mask)

    dump_errors(pred_types, ref_types, "pred_types_failures.xlsx", test_meta_types, ccode, src_names_types, functions, body_not_in_train_mask)
    dump_errors(pred_types, ref_types, "pred_types_unk.xlsx", test_meta_types, ccode, src_names_types, functions, is_unk_mask)
    dump_errors(pred_types, ref_types, "pred_types_empty.xlsx", test_meta_types, ccode, src_names_types, functions, is_empty_mask)
    # #dump_errors(pred_names, ref_names, "pred_name_failures.xlsx", test_meta_names, ccode, src_names, body_not_in_train_mask)

    # to_excel(type_not_in_train, "pred_types_common.xlsx")
    # to_excel(name_not_in_train, "pred_names_common.xlsx")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    results = json.load(open(args.pred_file))
    dataset = load_data(args.config_file)

    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    evaluate(dataset, results)