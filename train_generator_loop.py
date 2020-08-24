import os
import argparse
import subprocess
from hparams import HP
from itertools import product


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='test_loop')
arguments = parser.parse_args()

exp_name = arguments.exp_name
hp_name = HP.keys()
hp_grid = HP.values()

tested_hp_names = list()
for n, g in zip(hp_name, hp_grid):
    if len(g) > 1:
        tested_hp_names.append(n)

all_hps = product(*hp_grid)
for hp in all_hps:
    # naming the experiment folder
    try:
        hp_dict = {n: p for n, p in zip(hp_name, hp)}
        if len(tested_hp_names) == 0:
            unique_hparam = 'default'
        else:
            unique_hparam = '-'.join(f'{name}{hp_dict[name]}' for name in tested_hp_names)
        saveroot = os.path.join(exp_name, unique_hparam)

        # formatting atoms
        valid_hp_name = list(hp_name)
        ind_atoms = valid_hp_name.index('atoms')
        if len(hp_dict['atoms']) == 0:
            valid_hp_name.remove('atoms')
            hp = hp[:ind_atoms] + hp[ind_atoms+1:]
        else:
            atoms = ' '.join([str(atom) for atom in hp[ind_atoms]])
            hp = hp[:ind_atoms] + (atoms, ) + hp[ind_atoms + 1:]

        args = ' '.join(f'--{name}={param}' for name, param in zip(valid_hp_name, hp)) + f' --saveroot={saveroot}'
        cmd = f'python train_generator.py {args}'
        print(cmd)
        process = subprocess.call(cmd, shell=True)
    except KeyboardInterrupt:
        print('Process interrupted, terminating...')
        break
