"""
Usage:
    infer.py [options] CONFIG_FILE INPUT_JSON

Options:
    -h --help                  Show this screen.
"""

from utils.ghidra_function import CollectedFunction
from utils.dataset import Example, Dataset
from utils.code_processing import canonicalize_code

from model.model import TypeReconstructionModel

import _jsonnet

from docopt import docopt
import json

from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import torch

def main(args):

    config = json.loads(_jsonnet.evaluate_file(args["CONFIG_FILE"]))
    model = TypeReconstructionModel(config)
    model.eval()

    json_dict = json.load(open(args["INPUT_JSON"], "r"))
    # print(json_dict)
    cf = CollectedFunction.from_json(json_dict)
    print(cf)

    example = Example.from_cf(
        cf, binary_file=args["INPUT_JSON"], max_stack_length=1024, max_type_size=1024
    )
    #print(example)

    assert example.is_valid_example

    canonical_code = canonicalize_code(example.raw_code)
    example.canonical_code = canonical_code
    #print(example.canonical_code)

    # Create a dummy Dataset so we can call .annotate
    dataset = Dataset(config["data"]["test_file"], config["data"])

    #print(f"example src: {example.source}")
    #print(f"example target: {example.target}")

    example = dataset._annotate(example)

    collated_example = dataset.collate_fn([example])
    collated_example, _garbage = collated_example
    #print(collated_example)

    #tensor = torch.tensor([collated_example])
    #print(tensor)

    #single_example_loader = DataLoader([collated_example], batch_size=1)

    #trainer = pl.Trainer()
    #wat = trainer.predict(model, single_example_loader)
    #print(wat)

    with torch.no_grad():
        output = model(collated_example)
    print(f"The model output is: {output}")


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
