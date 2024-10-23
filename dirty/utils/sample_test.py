# Sample a few test cases from the prediction json file.

import json
import hjson
import argparse
import multiprocessing
import random
import json
import tempfile
import os
import shutil
import subprocess
import traceback
import tqdm
from utils.vocab import Vocab, UNKNOWN_ID

BIN_DIR = "/home/ed/Projects/DIRTY/dirt-binaries/all-binaries"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sample a few test cases from the dataset.")
    parser.add_argument("-n", "--num_samples", type=int, default=50, help="Number of samples to be generated")

    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("-o", "--output_file", help="Path to the output JSON file")
    parser.add_argument("--vocab", help="Path to the vocab file", default="data1/vocab.bpe10000")
    args = parser.parse_args()
    json_file = args.json_file

    with open(json_file, "r") as f:
        data = json.load(f)

    sampled_bins = random.sample(list(data.keys()), args.num_samples)

    sampled_functions = list((bin,random.choice(list(data[bin].items()))) for bin in sampled_bins)

    vocab = Vocab.load(args.vocab)

    # if args.output_file:
    #     with open(args.output_file, "w") as f:
    #         json.dump(dict(sampled_functions), f, indent=4)
    # else:
    #     print(json.dumps(dict(sampled_functions), indent=4), end="\n")

    def worker(input):
        bin, (func, jsondata) = input
        assert func.startswith("FUN_"), f"Function name {func} does not start with FUN_"
        funcaddr = func.replace("FUN_", "0x")
        #print(f"Bin: {bin}, Function: {func}")

        temp_dir = tempfile.mkdtemp()
        #print(f"Temporary directory: {temp_dir}")

        temp_json_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        temp_json_file_name = temp_json_file.name
        temp_json_file.close()

        # XXX Capture logs

        symbol_log = subprocess.check_output([
            "%s/support/analyzeHeadless" % os.environ["GHIDRA_INSTALL_DIR"],
            temp_dir,
            "DummyProject",
            "-readOnly",
            "-import",
            os.path.join(BIN_DIR, bin),
            "-postScript",
            "/home/ed/Projects/DIRTY/DIRTY-ghidra/scripts/DIRTY_eval.py",
            "/home/ed/Projects/DIRTY/DIRTY-ghidra/dirty/pred_mt.json",
            bin,
            funcaddr,
            temp_json_file_name
        ], stderr=subprocess.STDOUT)

        try:
            with open(temp_json_file_name, "r") as temp_json_file:
                symbol_data = json.loads(temp_json_file.read())
        except Exception as e:
            return {"exception": str(e), "log": symbol_log.decode("utf-8")}

        # Nothing to rewrite
        del symbol_data['rewritten_decompilation']

        #print("temp data: %s", str(temp_data))

        # Copy the binary to a temporary file name
        temp_bin_file = tempfile.NamedTemporaryFile(delete=False)
        temp_bin_file_name = temp_bin_file.name
        temp_bin_file.close()
        shutil.copyfile(os.path.join(BIN_DIR, bin), temp_bin_file_name)

        # Strip the binary
        subprocess.check_call(["strip", temp_bin_file_name])

        # print(f"Stripped binary: {temp_bin_file_name}")

        # Delete the json so we don't accidentally use the old results
        os.truncate(temp_json_file_name, 0)

        # Run Ghidra on the stripped version
        # XXX Capture logs
        strip_log = subprocess.check_output([
            "%s/support/analyzeHeadless" % os.environ["GHIDRA_INSTALL_DIR"],
            temp_dir,
            "DummyProject2",
            "-readOnly",
            "-import",
            temp_bin_file_name,
            "-postScript",
            "/home/ed/Projects/DIRTY/DIRTY-ghidra/scripts/DIRTY_eval.py",
            "/home/ed/Projects/DIRTY/DIRTY-ghidra/dirty/pred_mt.json",
            bin,
            funcaddr,
            temp_json_file_name
        ], stderr=subprocess.STDOUT)

        try:
            with open(temp_json_file_name, "r") as temp_json_file:
                strip_data = json.loads(temp_json_file.read())
        except Exception as e:
            return {"exception": str(e), "log": strip_log.decode("utf-8")}

        # Delete the temporary binary file
        os.remove(temp_bin_file_name)

        # Delete the temporary file
        os.remove(temp_json_file_name)

        d = {"strip": strip_data, "symbol": symbol_data}

        d["bin"] = bin
        d["func"] = func

        d["strip"]["aligned_vars"] = list(set(d["strip"]["vars"]) & set(jsondata.keys()))

        d["strip"]["aligned_frac"] = len(d["strip"]["aligned_vars"]) / len(d["strip"]["vars"])

        d["symbol"]["unknown_var_names"] = [v for v in d["symbol"]["vars"] if vocab.names[v] == UNKNOWN_ID]

        # d["symbol"]["origvars"] = [v[1] for v in jsondata.values()]

        # d["symbol"]["aligned_vars"] = list(set(d["symbol"]["vars"]) & {v[1] for v in jsondata.values()})
        # d["symbol"]["aligned_frac"] = len(d["symbol"]["aligned_vars"]) / len(d["symbol"]["vars"])

        if False:
            d["strip"]["log"] = strip_log.decode("utf-8")
            d["symbol"]["log"] = symbol_log.decode("utf-8")

        d["strip"]["aligned_frac"] = len(d["strip"]["aligned_vars"]) / len(d["strip"]["vars"])

        d["predictions"] = jsondata

        d["name_predictions"] = {k: v for k, v in jsondata.items() if v[1] != "<unk>" and k != v[1]}
        d["type_predictions"] = {k: v for k, v in jsondata.items() if v[0] != "<unk>"}

        return d
    
    def worker_catch(args):
        try:
            return worker(args)
        except Exception as e:
            traceback.print_exc()

            return json.dumps({"exception": str(e)})

    with multiprocessing.Pool(4) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(worker_catch, list(sampled_functions)), total=args.num_samples))

    total_variables_wo_symbols = sum(len(ex['strip']['vars']) for ex in results if "exception" not in ex) + 0.01
    total_variables_w_symbols = sum(len(ex['symbol']['vars']) for ex in results if "exception" not in ex) + 0.01
    #total_predictions = sum(len(ex['predictions']) for ex in results if "exception" not in ex)
    total_name_predictions = sum(len(ex['name_predictions']) for ex in results if "exception" not in ex)
    total_type_predictions = sum(len(ex['type_predictions']) for ex in results if "exception" not in ex)

    results = {"examples": results}
    results["total_variables_wo_symbols"] = total_variables_wo_symbols
    results["total_variables_w_symbols"] = total_variables_w_symbols

    #results["total_predictions"] = total_predictions
    results["total_name_predictions"] = total_name_predictions
    results["total_type_predictions"] = total_type_predictions

    #results["total_prediction_frac"] = total_predictions / total_variables_wo_symbols
    results["total_name_prediction_frac"] = total_name_predictions / total_variables_wo_symbols
    results["total_type_prediction_frac"] = total_type_predictions / total_variables_wo_symbols

    results["total_valid_ex"] = len([ex for ex in results["examples"] if "exception" not in ex]) 

    if args.output_file:
        with open(args.output_file, "w") as f:
            hjson.dump(results, f, indent=2)
    else:
        print(hjson.dumps(results, indent=2), end="\n")
