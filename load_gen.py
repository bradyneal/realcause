import os
import json
import torch
from train_generator import get_args, main


def load_gen(saveroot='save', dataroot=None):
    args = get_args()
    args_path = os.path.join(saveroot, 'args.txt')
    args.__dict__.update(json.load(open(args_path, 'r')))
    print(args)

    # overwriting args
    args.train = False
    args.eval = False
    args.comet = False
    args.saveroot = saveroot
    if dataroot is not None:
        args.dataroot = dataroot

    # initializing model
    model = main(args, False, False)

    # loading model params
    state_dicts = torch.load(model.savepath)
    for state_dict, net in zip(state_dicts, model.networks):
        net.load_state_dict(state_dict)
    return model, args
