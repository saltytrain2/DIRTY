import json
import tqdm
import argparse

class VariablePrediction:
    def __init__(
        self,
        name: str,
        code_tokens: str,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
    ):
        self.name = name
        self.code_tokens = code_tokens
        self.source_name = source_name
        self.source_type = source_type
        self.target_name = target_name
        self.target_type = target_type

def gen_errors(dataset, results):
    for example in tqdm(dataset):
        for src_name, tgt_name, tgt_type in zip(
            example.src_var_names, example.tgt_var_names, example.tgt_var_types
        ):
            pred_type, pred_name = (
                results.get(example.binary, {})
                .get(example.name, {})
                .get(src_name[2:-2], ("", ""))
            )


    pass


def main():
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    results = json.load(open(args.pred_file))
    dataset = load_data(args.config_file)
    
    import torch
    dataset = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=None)
    gen_errors(dataset, results)
    pass

if __name__ == "__main__":
    main()