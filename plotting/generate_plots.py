from pathlib import Path
import os
import zipfile
import json
from addict import Dict
from train_generator import get_data
from loading import load_gen
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm


def generate_plots(
    checkpoint_dir="./GenModelCkpts",
    #checkpoint_dir="./LinearModelCkpts",
    data_filter=None,
    plot_dir="./plots",
):

    checkpoint_dir = Path(checkpoint_dir).resolve()
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True, parents=True)
    dataset_roots = [Path(i) for i in os.listdir(checkpoint_dir)]
    # For each overall dataset (LBIDD, lalonde, etc.)
    for root in dataset_roots:
        root_plot_dir = plot_dir / root
        root_plot_dir.mkdir(exist_ok=True, parents=True)

        subdatasets = os.listdir(checkpoint_dir / root)

        if data_filter is not None:
            if data_filter not in str(root):
                continue

        # For each subdataset (psid1, cps1, etc.)
        for subdata in subdatasets:

            #if 'cps' not in subdata:
            #    continue

            subdata_plot_dir = root_plot_dir / subdata
            subdata_plot_dir.mkdir(exist_ok=True, parents=True)

            subdata_path = checkpoint_dir / root / subdata

            # Check if unzipping is necessary
            if (
                len(os.listdir(subdata_path)) == 1
                and ".zip" in os.listdir(subdata_path)[0]
            ):
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

            model, args = load_gen(saveroot=str(args.saveroot), dataroot="./datasets")

            ites, ate, w, t, y = get_data(args)

            plots = model.plot_ty_dists(verbose=False, title=False, transformed=True)

            for i, plot in enumerate(plots):
                if plot._suptitle is not None:
                    title = plot._suptitle.get_text()
                else:
                    title = str(i)
                plot.savefig(str(subdata_plot_dir / title) + ".png")


if __name__ == "__main__":
    generate_plots()