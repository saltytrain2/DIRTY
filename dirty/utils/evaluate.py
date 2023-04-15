import argparse
import json

import _jsonnet
import numpy as np
import wandb
from tqdm import tqdm

from utils.dataset import Dataset


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


def acc(preds, results, test_metas=None):
    return (preds == results).mean()


def mask_acc(preds, results, mask):
    return (preds[mask] == results[mask]).mean()


def body_in_train_acc(preds, results, test_metas):
    body_in_train_mask = np.array(
        [test_meta["function_body_in_train"] for test_meta in test_metas]
    )
    return mask_acc(preds, results, body_in_train_mask)


def body_not_in_train_acc(preds, results, test_metas):
    body_in_train_mask = np.array(
        [test_meta["function_body_in_train"] for test_meta in test_metas]
    )
    return mask_acc(preds, results, ~body_in_train_mask)


def struct_acc(preds, results, test_metas):
    struct_mask = np.array([test_meta["is_struct"] for test_meta in test_metas])
    return mask_acc(preds, results, struct_mask)


def struct_body_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["is_struct"] and test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def struct_body_not_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["is_struct"] and not test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def no_disappear_acc(preds, results, test_metas):
    no_disappear_mask = np.array([not test_meta["is_disappear"] for test_meta in test_metas])
    return mask_acc(preds, results, no_disappear_mask)


def no_disappear_body_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            not test_meta["is_disappear"] and test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def no_disappear_body_not_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            not test_meta["is_disappear"] and not test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def only_disappear_acc(preds, results, test_metas):
    no_disappear_mask = np.array([test_meta["is_disappear"] for test_meta in test_metas])
    return mask_acc(preds, results, no_disappear_mask)


def only_disappear_body_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["is_disappear"] and test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def only_disappear_body_not_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["is_disappear"] and not test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def func_all_disappear_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["func_all_disappear"] for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def func_all_disappear_body_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["func_all_disappear"] and test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def func_all_disappear_body_not_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["func_all_disappear"] and not test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def func_no_disappear_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["func_no_disappear"] for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def func_no_disappear_body_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["func_no_disappear"] and test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

def func_no_disappear_body_not_in_train_acc(preds, results, test_metas):
    mask = np.array(
        [
            test_meta["func_no_disappear"] and not test_meta["function_body_in_train"]
            for test_meta in test_metas
        ]
    )
    return mask_acc(preds, results, mask)

TYPE_METRICS = {
    "acc": acc,
    "body_in_train_acc": body_in_train_acc,
    "body_not_in_train_acc": body_not_in_train_acc,
    "struct_acc": struct_acc,
    "struct_body_in_train_acc": struct_body_in_train_acc,
    "struct_body_not_in_train_acc": struct_body_not_in_train_acc,
    "no_disappear_acc": no_disappear_acc,
    "no_disappear_body_in_train_acc": no_disappear_body_in_train_acc,
    "no_disappear_body_not_in_train_acc": no_disappear_body_not_in_train_acc,
    "only_disappear_acc": only_disappear_acc,
    "only_disappear_body_in_train_acc": only_disappear_body_in_train_acc,
    "only_disappear_body_not_int_train_acc": only_disappear_body_not_in_train_acc,
    "func_no_disappear_acc": func_no_disappear_acc,
    "func_no_disappear_body_in_train_acc": func_no_disappear_body_in_train_acc,
    "func_no_disappear_body_not_int_train_acc": func_no_disappear_body_not_in_train_acc,
    "func_all_disappear_acc": func_all_disappear_acc,
    "func_all_disappear_body_in_train_acc": func_all_disappear_body_in_train_acc,
    "func_all_disappear_body_not_int_train_acc": func_all_disappear_body_not_in_train_acc,
}

NAME_METRICS = {
    "accuracy": acc,
    "body_in_train_acc": body_in_train_acc,
    "body_not_in_train_acc": body_not_in_train_acc,
    "no_disappear_acc": no_disappear_acc,
    "no_disappear_body_in_train_acc": no_disappear_body_in_train_acc,
    "no_disappear_body_not_in_train_acc": no_disappear_body_not_in_train_acc,
    "only_disappear_acc": only_disappear_acc,
    "only_disappear_body_in_train_acc": only_disappear_body_in_train_acc,
    "only_disappear_body_not_int_train_acc": only_disappear_body_not_in_train_acc,
}


def evaluate(dataset, results, type_metrics, name_metrics):
    pred_names, ref_names, pred_types, ref_types = [], [], [], []
    test_meta_types, test_meta_names = [], []
    examples_w_structs = []
    num_functions, num_all_disappear, num_no_disappear = 0, 0, 0
    for example in tqdm(dataset):
        # one example is one function: check if all variables are disappear or if all variables are actual variables
        all_disappear = True
        no_disappear = True

        for tgt_type in example.tgt_var_types_str:
            if dataset.dataset.vocab.types.id2word[dataset.dataset.vocab.types[tgt_type]] == "disappear":
                no_disappear = False
            else:
                all_disappear = False

        num_functions += 1
        if all_disappear:
            num_all_disappear += 1
        if no_disappear:
            num_no_disappear += 1

        for src_name, tgt_name, tgt_type in zip(
            example.src_var_names, example.tgt_var_names, example.tgt_var_types_str
        ):
            pred_type, _ = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )
            pred_types.append(pred_type)
            ref_types.append(tgt_type)
            test_meta = example.test_meta.copy()
            test_meta["is_struct"] = dataset.dataset.vocab.types.id2word[
                dataset.dataset.vocab.types[tgt_type]
            ].startswith("struct ")
            if test_meta["is_struct"]:
                examples_w_structs.append(example.binary)

            test_meta["is_disappear"] = dataset.dataset.vocab.types.id2word[
                dataset.dataset.vocab.types[tgt_type]
            ].startswith("disappear")

            test_meta["func_all_disappear"] = all_disappear
            test_meta["func_no_disappear"] = no_disappear
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

    struct_counter = 0
    disappear_counter = 0
    for test_meta in test_meta_types:
        struct_counter += 1 if test_meta["is_struct"] else 0
        disappear_counter += 1 if test_meta["is_disappear"] else 0

    with open("struct_files.txt", "w") as file:
        for elem in examples_w_structs:
            file.write(str(elem) + "\n")

    wandb.log(
        {
            "total variables": len(test_meta_types),
            "num structs": struct_counter,
            "num disappear": disappear_counter
        }
    )

    wandb.log(
        {
            "total examples:": num_functions,
            "examples all disappear": num_all_disappear,
            "examples no disappear": num_no_disappear
        }
    )
    
    for metric_name, metric in type_metrics.items():
        wandb.log(
            {
                f"test_retype_{metric_name}": metric(
                    pred_types, ref_types, test_meta_types
                )
            }
        )

    pred_names = np.array(pred_names, dtype=object)
    ref_names = np.array(ref_names, dtype=object)

    for metric_name, metric in name_metrics.items():
        wandb.log(
            {
                f"test_rename_{metric_name}": metric(
                    pred_names, ref_names, test_meta_names
                )
            }
        )


#     mt_evaluate(dataset, results)

# def mt_evaluate(dataset, results):
#     mt_results = json.load(open("pred_mt.json"))
#     pred_names, ref_names, pred_types, ref_types = [], [], [], []
#     for example in tqdm(dataset):
#         for src_name, tgt_name, tgt_type in zip(example.src_var_names, example.tgt_var_names, example.tgt_var_types_str):
#             pred_type, _ = results.get(example.binary, {}).get(example.name, {}).get(src_name[2:-2], ("", ""))
#             _, mt_pred_name = mt_results.get(example.binary, {}).get(example.name, {}).get(src_name[2:-2], ("", ""))
#             if mt_pred_name == tgt_name[2:-2] and tgt_name != "@@@@":
#                 pred_types.append(pred_type)
#                 ref_types.append(tgt_type)
#             if src_name != tgt_name and tgt_name != "@@@@":
#                 _, pred_name = results.get(example.binary, {}).get(example.name, {}).get(src_name[2:-2], ("", ""))
#                 mt_pred_type, _ = mt_results.get(example.binary, {}).get(example.name, {}).get(src_name[2:-2], ("", ""))
#                 if mt_pred_type == tgt_type:
#                     pred_names.append(pred_name)
#                     ref_names.append(tgt_name[2:-2])
#     pred_types = np.array(pred_types, dtype=object)
#     ref_types = np.array(ref_types, dtype=object)

#     pred_names = np.array(pred_names, dtype=object)
#     ref_names = np.array(ref_names, dtype=object)

#     wandb.log({"test_rnonrt_accuracy": acc(pred_names, ref_names)})
#     wandb.log({"test_rtonrn_accuracy": acc(pred_types, ref_types)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    results = json.load(open(args.pred_file))
    dataset = load_data(args.config_file)
    import torch

    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)

    wandb.init(name=f"test_{args.pred_file}", project="dire")
    evaluate(dataset, results, TYPE_METRICS, NAME_METRICS)
