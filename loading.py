import os
import json
import zipfile
from pathlib import Path
import pandas as pd

import torch
from addict import Dict

from train_generator import get_args, main
from consts import REALCAUSE_DATASETS_FOLDER, N_SAMPLE_SEEDS


def load_realcause_dataset(dataset, sample=0):
    valid_datasets = {'lalonde_cps', 'lalonde_psid', 'twins'}
    dataset = dataset.lower()
    if dataset not in valid_datasets:
        raise ValueError('Invalid dataset "{}" ... Valid datasets: {}'
                         .format(dataset, valid_datasets))
    if not isinstance(sample, int):
        raise ValueError('sample must be an integer')
    if 0 < sample >= N_SAMPLE_SEEDS:
        raise ValueError('sample must be between 0 and {}'
                         .format(N_SAMPLE_SEEDS - 1))

    dataset_file = Path(REALCAUSE_DATASETS_FOLDER) / '{}_sample{}.csv'.format(dataset, sample)
    return pd.read_csv(dataset_file)


def load_gen(saveroot='save', dataroot=None):
    args = get_args().parse_args([])
    args_path = os.path.join(saveroot, 'args.txt')
    args.__dict__.update(json.load(open(args_path, 'r')))
    print(args)

    # overwriting args
    args.train = False
    args.eval = False
    args.comet = False
    args.saveroot = saveroot
    args.comet = False
    if dataroot is not None:
        args.dataroot = dataroot

    # initializing model
    model = main(args, False, False)

    # loading model params
    state_dicts = torch.load(model.savepath)
    for state_dict, net in zip(state_dicts, model.networks):
        net.load_state_dict(state_dict)
    return model, args


def load_from_folder(dataset, checkpoint_path="./GenModelCkpts.zip"):
    checkpoint_path = Path(checkpoint_path).resolve()
    root_dir = checkpoint_path.parent
    checkpoint_dir = root_dir / checkpoint_path.stem
    if not checkpoint_dir.is_dir():
        with zipfile.ZipFile(checkpoint_path, "r") as zip_ref:
            zip_ref.extractall()

    dataset_roots = os.listdir(checkpoint_dir)
    dataset_stem = dataset.split("_")[0]
    subdata_stem = dataset.split("_")[-1]

    assert dataset_stem in dataset_roots
    subdatasets = os.listdir(checkpoint_dir / dataset_stem)
    assert subdata_stem in subdatasets

    subdata_path = checkpoint_dir / Path(dataset_stem) / Path(subdata_stem)
    # Check if unzipping is necessary
    if len(os.listdir(subdata_path)) == 1 and ".zip" in os.listdir(subdata_path)[0]:
        zip_name = os.listdir(subdata_path)[0]
        zip_path = subdata_path / zip_name
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(subdata_path)

    subfolders = [f.path for f in os.scandir(subdata_path) if f.is_dir()]
    assert len(subfolders) == 1

    model_folder = subdata_path / Path(subfolders[0])

    with open(model_folder / "args.txt") as f:
        args = Dict(json.load(f))

    args.saveroot = model_folder
    args.dataroot = "./datasets/"
    args.comet = False

    # Now load model
    model, args = load_gen(saveroot=str(args.saveroot), dataroot="./datasets")

    return model, args