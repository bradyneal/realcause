import os
import argparse
import subprocess
from hparams import HP
from itertools import product
from multiprocessing import Pool


def run_exp(hp):
    # naming the experiment folder
    hp_dict = {name: p for name, p in zip(hp_name, hp)}
    if len(tested_hp_names) == 0:
        unique_hparam = "default"
    else:
        unique_hparam = list()
        for name in tested_hp_names:
            param = hp_dict[name]
            if isinstance(param, list):
                unique_hparam.append(f"{name}{'+'.join(str(p) for p in param)}")
            else:
                unique_hparam.append(f"{name}{param}")
        unique_hparam = "-".join(unique_hparam)
    saveroot = os.path.join(exp_name, unique_hparam)

    # formatting atoms
    valid_hp_name = list(hp_name)
    ind_atoms = valid_hp_name.index("atoms")
    if len(hp_dict["atoms"]) == 0:
        valid_hp_name.remove("atoms")
        hp = hp[:ind_atoms] + hp[ind_atoms+1:]
    else:
        atoms = " ".join([str(atom) for atom in hp[ind_atoms]])
        hp = hp[:ind_atoms] + (atoms,) + hp[ind_atoms+1:]

    # formatting dist_args
    ind_dist_args = valid_hp_name.index("dist_args")
    if len(hp_dict["dist_args"]) == 0:
        valid_hp_name.remove("dist_args")
        hp = hp[:ind_dist_args] + hp[ind_dist_args + 1:]
    else:
        dist_args = " ".join([str(dist_arg) for dist_arg in hp[ind_dist_args]])
        hp = hp[:ind_dist_args] + (dist_args,) + hp[ind_dist_args + 1:]

    args = (
        " ".join(f"--{name} {param}" for name, param in zip(valid_hp_name, hp))
        + f" --saveroot={saveroot}"
    )
    cmd = f"python train_generator.py {args}"
    _ = subprocess.call(cmd, shell=True)

    print(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test_loop")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of cores")

    arguments = parser.parse_args()

    exp_name = arguments.exp_name
    hp_name = HP.keys()
    hp_grid = HP.values()

    tested_hp_names = list()
    for n, g in zip(hp_name, hp_grid):
        if len(g) > 1:
            tested_hp_names.append(n)

    all_hps = product(*hp_grid)

    pool = Pool(arguments.num_workers)  # Create a multiprocessing Pool
    pool.map(run_exp, all_hps)  # process data_inputs iterable with pool
