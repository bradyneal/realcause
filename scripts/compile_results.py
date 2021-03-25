# import csv
from pathlib import Path
import os
import ast
import pandas as pd
from tqdm import tqdm
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filters", nargs="+", help="filter experiments (e.g. `linear dim_h128`)"
    )
    parser.add_argument(
        "--exp_dir", type=str, help="Path to experiment folder", required=True
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    arguments = parser.parse_args()

    filter_criterion = arguments.filters

    experiment_dir = Path(arguments.exp_dir)
    experiments = os.listdir(experiment_dir)

    results = {}
    for exp in tqdm(experiments):
        if filter_criterion is not None:
            for word in filter_criterion:
                if word not in exp:
                    if arguments.verbose:
                        print(f"Omitting exp {exp}")
                    continue

        exp = Path(exp)

        if (experiment_dir / exp / "summary.txt").is_file():
            with open(experiment_dir / exp / "summary.txt") as f:
                exp_contents = f.read()
                if "NaN" in exp_contents or "nan" in exp_contents:
                    continue
                exp_dict = ast.literal_eval(exp_contents)
                results[exp] = exp_dict

    df = pd.DataFrame.from_dict(results).transpose()
    df.to_csv(experiment_dir.stem + ".csv")
