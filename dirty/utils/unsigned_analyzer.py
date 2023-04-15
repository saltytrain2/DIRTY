import re
import argparse
import json
import subprocess
import time
import _jsonnet
from tqdm import tqdm
from collections import defaultdict
import pickle
import multiprocessing
from dataclasses import dataclass
import random
import numpy as np
import os

from utils.dataset import Dataset, Example

# magic sets for integer types and select assembly operations
UNSIGNED_INT_TYPES = {"uint", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "size_t", "ulong", "uchar",
                      "uint32", "uint16", "uint8", "uint64", "UINT32", "UINT16", "UINT8", "UINT64", 
                      "u_int", "__u32", "__u16", "__u8", "__u64", "u32", "u16", "u8", "u64", "U32",
                      "U16", "U8", "U64"}

SIGNED_INT_TYPES = {"int", "int8_t", "int16_t", "int32_t", "int64_t", "long", "long long", "short", "char",
                    "int32", "int16", "int8", "int64", "INT32", "INT16", "INT8", "INT64", "INTEGER", "s32",
                    "s16", "s8", "s64", "S32", "S16", "S8", "S64", "__s32", "__s16", "__s8", "__s64"}

ALL_INTS = UNSIGNED_INT_TYPES.union(SIGNED_INT_TYPES)

RESTRICT_ARITH_OPS = {"add", "sub", "mul", "imul", "div", "idiv", "inc", "dec"}
RESTRICT_LOGIC_OPS = {"and", "or", "xor", "not", "shl", "shr", "sal", "sar"}
RESTRICT_JUMPS_OPS = {"jo", "jno", "js", "jns", "je", "jz", "jne", "jnz", "jb", "jnae", "jc", "jnb", "jae", "jnc", "jbe", "jna", "ja", "jnbe", "jl", "jnge", "jge", "jnl", "jle", "jng", "jg", "jnle", "jp", "jpe", "jnp", "jpo", "jcxz", "jecxz"}

RESTRICTED_ASM = RESTRICT_ARITH_OPS.union(RESTRICT_LOGIC_OPS).union(RESTRICT_JUMPS_OPS)
RESTRICTED_ASM_REGEX = re.compile("|".join(["(" + op + "\s)" for op in list(RESTRICTED_ASM)]))

class AsmFunction:
    def __init__(self, function_name, code):
        self._function_name = function_name
        self._code = code
        self._tokens = []
        self._unary = {"pushq", "popq", "inc", "dec"}
        self._noop = {"nop", "leaveq", "retq"}
        self._locations = {}
        #self._is_unsigned = {}
        self._is_unsigned = False

    def lex_function(self) -> None:
        # regex_match = re.findall("[a-z0-9\-%()]+", self._code)
        regex_match = re.finditer(RESTRICTED_ASM_REGEX, self._code)

        if regex_match is None:
            raise RuntimeError("No function body found")
        
        self._tokens = regex_match
        pass


    def track_lifetimes(self, op_dict):
        regex_match = re.finditer(RESTRICTED_ASM_REGEX, self._code)
    
        for match in regex_match:
            op_dict[match.group(0).strip()] += 1

        return regex_match is not None

def add_options(parser):
    parser.add_argument(
        "-b", "--binaries-dir", type=str, required=True, help="Path to dataset binaries"
    )
    parser.add_argument(
        "--pred-file", type=str, required=True, help="Saved predictions on a dataset"
    )
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--init", action="store_true", default=False)


def load_data(config_file):
    config = json.loads(_jsonnet.evaluate_file(config_file))["data"]
    config["max_num_var"] = 1 << 30
    dataset = Dataset(config["train_file"], config)
    return dataset

def load_testdata(config_file):
    config = json.loads(_jsonnet.evaluate_file(config_file))["data"]
    config["max_num_var"] = 1 << 30
    dataset = Dataset(config["test_file"], config)
    return dataset

def create_asm_regex():
    return re.compile(f"<(?P<fun_name>\w+)>:(?P<code>(.|\n)*?(\n\n|$))")

def create_asm_regex_fun(fun_name):
    fun_name = fun_name.replace("[]", "\\[]")
    fun_name = fun_name.replace("()", "\\(\\)")
    return f"<(?P<fun_name>{fun_name})>:(?P<code>(.|\n)*?(\n\n|$))"

def create_objdump_script(path_to_binary):
    return f"objdump -d --no-show-raw-insn {path_to_binary} | sed -r -e \"s/[a-f0-9]+://g\""

def int_mask(preds, exclude_int = False):
    return np.array([t in ALL_INTS and (t != "int" if exclude_int else True) for t in preds])

def unsigned_mask(preds):
    return np.array([t in UNSIGNED_INT_TYPES for t in preds])

def not_in_train_mask(test_metas):
    return np.array([not test_meta["function_body_in_train"] for test_meta in test_metas])

@dataclass
class Param:
    example: Example
    binary_dir: str
    unsigned_op_counts: dict
    signed_op_counts: dict

def extract_asm_percentages(param : Param):
    src_types = {src_type for src_type in param.example.src_var_types_str}

    if src_types.intersection(ALL_INTS):
        process = subprocess.run(create_objdump_script(param.binary_dir + param.example.binary), shell=True, capture_output=True, encoding="utf-8")
        if process.returncode:
            return
            
        bin_str = process.stdout
        regex_match = re.search(create_asm_regex_fun(param.example.name), bin_str)
        if regex_match is None:
            return
        
        function = AsmFunction(param.example.name, regex_match.group("code"))
        function.track_lifetimes(param.unsigned_op_counts if src_types.intersection(UNSIGNED_INT_TYPES) else param.signed_op_counts)

def merge_dicts(dicts):
    output_dict = {}
    for opcode in RESTRICTED_ASM:
        output_dict[opcode] = 0

    for d in dicts:
        for key, value in d.items():
            output_dict[key] += value
    
    return output_dict

# anonymous lambda to print value to string
def evaluate(dataset, binaries_dir):
    num_processes = 16

    manager = multiprocessing.Manager()
    unsigned_op_counts = [manager.dict() for _ in range(int(num_processes))]
    signed_op_counts = [manager.dict() for _ in range(int(num_processes))]

    for opcode in RESTRICTED_ASM:
        for unsigned, signed in zip(unsigned_op_counts, signed_op_counts):
            unsigned[opcode] = 0
            signed[opcode] = 0

    params = [Param(example, binaries_dir, random.choice(unsigned_op_counts), random.choice(signed_op_counts)) for example in dataset]

    with multiprocessing.Pool(processes=16) as p:
        for _ in tqdm(p.imap_unordered(extract_asm_percentages, params, 25)):
            pass

    dicts = (merge_dicts(unsigned_op_counts), merge_dicts(signed_op_counts))
    pickle.dump(dicts, open("asm_opcounts.p", "wb"))

def avg_dict(d):
    tot = sum(d.values())
    return {k : (v / tot) for k, v in d.items()}

def load_opcount_dicts():
    if not os.path.exists("asm_opcounts.p"):
        raise RuntimeError("Please run this file in the dirty folder using python submodules")

    unsigned_counts, signed_counts = pickle.load(open("asm_opcounts.p", "rb"))

    unsigned_counts = avg_dict(unsigned_counts)
    signed_counts = avg_dict(signed_counts)

    return (unsigned_counts, signed_counts)

def extract_function_opcode_counts(binary, fun_name, binary_dir):
    process = subprocess.run(create_objdump_script(binary_dir + binary), shell=True, capture_output=True, encoding="utf-8")
    if process.returncode:
        return {}
    
    bin_str = process.stdout
    regex_match = re.search(create_asm_regex_fun(fun_name), bin_str)
    if regex_match is None:
        return {}

    function = AsmFunction(fun_name, regex_match.group("code"))
    res = defaultdict(int)
    function.track_lifetimes(res)
    return dict(res)

    pass

def chisq(opcounts, ref):
    observed = lambda x : opcounts[x] if x in opcounts else 0
    return sum([(v - observed(k)) ** 2 for k, v in ref.items()])

def augment(pred_types, ref_types, binaries, fun_names, binary_dir):
    pred_unsigned = unsigned_mask(pred_types)
    ref_unsigned = unsigned_mask(ref_types)

    unsigned_ref_opcounts, signed_ref_opcounts = load_opcount_dicts()

    for i, binary, fun_name in zip(range(len(pred_unsigned)), binaries, fun_names):
        op_counts = extract_function_opcode_counts(binary, fun_name, binary_dir)
        op_counts = avg_dict(op_counts)
        unsigned_chisq = chisq(op_counts, unsigned_ref_opcounts)
        signed_chisq = chisq(op_counts, signed_ref_opcounts)
        if pred_unsigned[i] and unsigned_chisq > signed_chisq and abs(unsigned_chisq - signed_chisq) > 0.1 * unsigned_chisq:
            pred_unsigned[i] = False
        elif not pred_unsigned[i] and signed_chisq > unsigned_chisq and abs(signed_chisq - unsigned_chisq) > 0.1 * signed_chisq:
            pred_unsigned[i] = True

    return np.logical_not(pred_unsigned ^ ref_unsigned).mean()
    pass

def retype(dataset, results, binary_dir):
    pred_types, ref_types, binaries, fun_names, test_metas = [], [], [], [], []

    for example in tqdm(dataset):
        for src_name, tgt_type in zip(example.src_var_names, example.tgt_var_types_str):
            pred_type, _ = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )
            pred_types.append(pred_type)
            ref_types.append(tgt_type)
            binaries.append(example.binary)
            fun_names.append(example.name)
            test_metas.append(example.test_meta.copy())

    
    pred_types = np.array(pred_types, dtype=object)
    ref_types  = np.array(ref_types, dtype=object)
    binaries = np.array(binaries, dtype=object)
    fun_names = np.array(fun_names, dtype=object)

    body_not_in_train_mask = not_in_train_mask(test_metas)
    pred_types = pred_types[body_not_in_train_mask]
    ref_types = ref_types[body_not_in_train_mask]
    binaries = binaries[body_not_in_train_mask]
    fun_names = fun_names[body_not_in_train_mask]

    pred_int_mask = int_mask(pred_types, True)
    pred_types = pred_types[pred_int_mask]
    ref_types = ref_types[pred_int_mask]
    binaries = binaries[pred_int_mask]
    fun_names = fun_names[pred_int_mask]

    regular_int_acc = np.logical_not(unsigned_mask(pred_types) ^ unsigned_mask(ref_types)).mean()
    augmented_int_acc = augment(pred_types, ref_types, binaries, fun_names, binary_dir)

    print(f"unmodified_acc:{regular_int_acc}")
    print(f"augmented_acc: {augmented_int_acc}")


def main():
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    binary_dir = args.binaries_dir
    dataset = load_data(args.config_file)
    test_dataset = load_testdata(args.config_file)
    results = json.load(open(args.pred_file))

    import torch
    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    if args.init:
        evaluate(dataset, binary_dir)

    retype(test_dataset, results, binary_dir)

if __name__ == "__main__":
    main()