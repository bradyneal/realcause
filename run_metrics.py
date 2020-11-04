from pathlib import Path
import os
import zipfile
import json
from addict import Dict
from train_generator import get_data
from load_gen import load_gen
import numpy as np
from collections import OrderedDict
import time
from tqdm import tqdm


def get_univariate_results(model, num_tests=100, verbose=False):
    all_runs = list()
    t_pvals = list()
    y_pvals = list()

    for _ in tqdm(range(num_tests)):
        uni_metrics = model.get_univariate_quant_metrics(
            dataset="test", verbose=verbose
        )
        all_runs.append(uni_metrics)
        t_pvals.append(uni_metrics["t_ks_pval"])
        y_pvals.append(uni_metrics["y_ks_pval"])

    summary = OrderedDict()
    summary.update(avg_t_pval=sum(t_pvals) / num_tests)
    summary.update(avg_y_pval=sum(y_pvals) / num_tests)

    return summary


def get_multivariate_results(model, include_w, num_tests=100, n=1000):
    # wasserstein1 pval', 'wasserstein2 pval', 'Friedman-Rafsky pval', 'kNN pval', 'Energy pval'
    w1_pval = list()
    w2_pval = list()
    fr_pval = list()
    knn_pval = list()
    energy_pval = list()

    for _ in tqdm(range(num_tests)):
        multi_metrics = model.get_multivariate_quant_metrics(
            dataset="test", n=n, include_w=include_w
        )
        w1_pval.append(multi_metrics["wasserstein1 pval"])
        w2_pval.append(multi_metrics["wasserstein2 pval"])
        fr_pval.append(multi_metrics["Friedman-Rafsky pval"])
        knn_pval.append(multi_metrics["kNN pval"])
        energy_pval.append(multi_metrics["Energy pval"])

    summary = OrderedDict()
    summary.update(avg_w1_pval=sum(w1_pval) / num_tests)
    summary.update(avg_w2_pval=sum(w2_pval) / num_tests)
    summary.update(avg_fr_pval=sum(fr_pval) / num_tests)
    summary.update(avg_knn_pval=sum(knn_pval) / num_tests)
    summary.update(avg_energy_pval=sum(energy_pval) / num_tests)

    return summary


def evaluate_directory(
    checkpoint_dir="./GenModelCkpts",
    data_filter=None,
    num_tests=100,
    n=1000,
    include_w=True,
    results_dir="./results",
):

    checkpoint_dir = Path(checkpoint_dir).resolve()
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    dataset_roots = [Path(i) for i in os.listdir(checkpoint_dir)]
    results = {}
    # For each overall dataset (LBIDD, lalonde, etc.)
    for root in dataset_roots:
        subdatasets = os.listdir(checkpoint_dir / root)

        if data_filter is not None:
            if data_filter not in str(root) or "1k" in str(root):
                continue

        # For each subdataset (psid1, cps1, etc.)
        for subdata in subdatasets:
            if "psid" not in subdata:
                continue

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

            ites, ate, w, t, y = get_data(args)
            print("BROH ate: ", ate)

            # ites = None
            # ate = None

            # Now load model
            model, args = load_gen(saveroot=str(args.saveroot), dataroot="./datasets")

            # TODO: compare the pipeline of noisy_ate() to ite() too see what's different
            if ate is not None:
                t0 = np.zeros((t.shape[0], 1))
                t1 = np.ones((t.shape[0], 1))
                print("computing ate...", end="\r", flush=True)
                noisy_ate = model.noisy_ate(w=w, t1=t1, t0=t0, transform_w=True)
            else:
                noisy_ate = None

            if ites is not None:
                print("computing ite estimate...", end="\r", flush=True)
                ite_est = model.ite(w=w, noisy=True)
                pehe = np.sqrt(np.median(np.square(ites - ite_est)))
            else:
                ite_est = None
                pehe = None

            print("computing uni metrics...", end="\r", flush=True)
            # uni_summary = get_univariate_results(model, num_tests=num_tests)
            uni_summary = None
            print("computing multi metrics include_w=True...", end="\r", flush=True)
            multi_summary_w = get_multivariate_results(
                model, num_tests=num_tests, n=n, include_w=True
            )
            print("computing multi metrics include_w=True...", end="\r", flush=True)
            multi_summary_no_w = get_multivariate_results(
                model, num_tests=num_tests, n=n, include_w=False
            )

            if args.test_size is None:
                total = args.train_prop + args.val_prop + args.test_prop
                n_total = y.shape[0]
                n_train = round(n_total * args.train_prop / total)
                n_val = round(n_total * args.val_prop / total)
                n_test = n_total - n_train - n_val
            else:
                n_test = args.test_size

            subdict = {}
            subdict["univariate_test_size"] = n_test
            subdict["multivariate_test_size"] = n
            subdict["pehe"] = pehe
            subdict["ate"] = ate
            subdict["ate_est"] = noisy_ate
            subdict["univariate_metrics"] = uni_summary
            subdict["multivariate_metrics_w"] = multi_summary_w
            subdict["multivariate_metrics_no_w"] = multi_summary_no_w

            results[str(root) + "_" + str(subdata)] = subdict

            if data_filter is not None:
                with open(
                    results_dir / (data_filter + "results.json"), "w"
                ) as fp:
                    json.dump(results, fp, indent=4)

            else:
                with open(results_dir / "results.json", "w") as fp:
                    json.dump(results, fp, indent=4)


def eval_ckpt(checkpoint_path):
    with open(checkpoint_path / "args.txt") as f:
        args = Dict(json.load(f))
        args.saveroot = checkpoint_path
        ites, ate, w, t, y = get_data(args)
        # Now load model
        model, args = load_gen(saveroot=str(args.saveroot), dataroot="./datasets")


if __name__ == "__main__":
    evaluate_directory(data_filter="lalonde", num_tests=1, n=200)
