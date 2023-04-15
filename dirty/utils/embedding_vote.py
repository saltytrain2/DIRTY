import argparse
import json
from collections import defaultdict
from dataclasses import dataclass

import _jsonnet
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import Tensor

from utils.dataset import Dataset
from model.model import TypeReconstructionModel
from utils.unsigned_analyzer import ALL_INTS, UNSIGNED_INT_TYPES, SIGNED_INT_TYPES

@dataclass
class Param:
    name: str
    embedding: Tensor
    is_unsigned: bool

def add_options(parser):
    parser.add_argument(
        "--pred-file", type=str, required=True, help="Saved predictions on a dataset"
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Configuration of model repo/settings"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model"
    )
    pass



def load_train_data(config_file):
    config = json.loads(_jsonnet.evaluate_file(config_file))["data"]
    config["max_num_var"] = 1 << 30
    return Dataset(config["train_file"], config)

def load_test_data(config_file):
    config = json.loads(_jsonnet.evaluate_file(config_file))["data"]
    config["max_num_var"] = 1 << 30
    return Dataset(config["train_file"], config)

def get_embeddings(model, batch):
    with torch.no_grad():
        input_dict, target_dict = batch
        # print(target_dict)
        temp = model.get_unmasked_logits(batch)
        return temp

def get_train_embeddings(args):
    config = json.loads(_jsonnet.evaluate_file(args.config_file))
    train_dataset = load_train_data(args.config_file)
    train_dataset = DataLoader(train_dataset, num_workers=8, collate_fn=Dataset.collate_fn, batch_size=5)
    model = TypeReconstructionModel(config)
    ckpt = torch.load(args.ckpt)
    # print(type(ckpt))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return [get_embeddings(model, batch) for batch in tqdm(train_dataset)]


def main():
    parser = argparse.ArgumentParser()
    add_options(parser)
    args = parser.parse_args()

    print(get_train_embeddings(args)[0])

if __name__ == "__main__":
    main()