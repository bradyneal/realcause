import os
import json
import torch
from train_generator import get_args, main


def load_gen(saveroot='save'):
    args = get_args()
    args_path = os.path.join(saveroot, 'args.txt')
    args.__dict__.update(json.load(open(args_path, 'r')))
    print(args)

    args.train = False
    args.eval = False
    model = main(args, False, False)
    state_dicts = torch.load(model.savepath)
    for state_dict, net in zip(state_dicts, model.networks):
        net.load_state_dict(state_dict)
    return model, args
