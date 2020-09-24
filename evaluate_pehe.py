from pathlib import Path
import os
import zipfile
import json
from addict import Dict
from train_generator import get_data
from load_gen import load_gen
import numpy as np

checkpoint_dir = Path("./GenModelCkpts").resolve()
dataset_roots = [Path(i) for i in os.listdir(checkpoint_dir)]
pehes = {}
# For each overall dataset (LBIDD, lalonde, etc.)
for root in dataset_roots:
    subdatasets = os.listdir(checkpoint_dir / root)
    # For each subdataset (psid1, cps1, etc.)
    for subdata in subdatasets:
        subdata_path = checkpoint_dir / root / subdata

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

        ites, ate, w, t, y = get_data(args)
        if "lbidd" not in args.data:
            continue

        # Now load model
        model, args = load_gen(saveroot=str(args.saveroot), dataroot="./datasets")
        print(f"Getting ites for dataset {args.data}")

        print("computing ite...", flush=True)
        # ite_est = np.zeros(ites.shape)
        ite_est = model.ite(w=w, noisy=True)

        # TODO: compare the pipeline of noisy_ate() to ite() too see what's different
        t0 = np.zeros((t.shape[0], 1))
        t1 = np.ones((t.shape[0], 1))
        noisy_ate = model.noisy_ate(w=w, t1=t1, t0=t0, transform_w=True)
        # noisy_ate = 0.0

        pehe = np.sqrt(np.mean(np.square(ites - ite_est)))
        subdict = {}
        subdict["pehe"] = pehe
        subdict["ate"] = ate
        subdict["ate_est"] = noisy_ate
        pehes[args.data] = subdict

with open("pehes.json", "w") as fp:
    json.dump(pehes, fp, indent=4)