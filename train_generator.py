from comet_ml import Experiment
import argparse
import os
import numpy as np
import torch
from data.lalonde import load_lalonde
from data.lbidd import load_lbidd
from data.ihdp import load_ihdp
from models import TarNet, preprocess, TrainingParams, MLPParams
from models import distributions
import helpers
from collections import OrderedDict
import json


def get_data(args):
    data_name = args.data.lower()
    ate = None
    if data_name == "lalonde":
        w, t, y = load_lalonde(dataroot=args.dataroot)
    elif data_name == "lalonde_rct":
        w, t, y = load_lalonde(rct=True, dataroot=args.dataroot)
    elif data_name == "lalonde_cps1":
        w, t, y = load_lalonde(obs_version="cps1", dataroot=args.dataroot)
    elif data_name.startswith("lbidd"):
        # Valid string formats: lbidd_<link>_<n> and lbidd_<link>_<n>_counterfactual
        # Valid <link> options: linear, quadratic, cubic, exp, and log
        # Valid <n> options: 1k, 2.5k, 5k, 10k, 25k, and 50k
        options = data_name.split("_")
        link = options[1]
        n = options[2]
        observe_counterfactuals = (len(options) == 4) and (options[3] == "counterfactual")
        d = load_lbidd(n=n, observe_counterfactuals=observe_counterfactuals, 
                       link=link, dataroot=args.dataroot, return_ate=True)
        ate = d["ate"]
        if observe_counterfactuals:
            w, t, y = d["obs_counterfactual_w"], d["obs_counterfactual_t"], d["obs_counterfactual_y"]
        else:
            w, t, y = d["w"], d["t"], d["y"]
    elif data_name == "ihdp":
        d = load_ihdp()
        w, t, y = d["w"], d["t"], d["y"]
    elif data_name == "ihdp_counterfactual":
        d = load_ihdp(observe_counterfactuals=True)
        w, t, y = d["w"], d["t"], d["y"]
    else:
        raise (Exception("dataset {} not implemented".format(args.data)))

    return ate, w, t, y


def get_distribution(args):
    """
    args.dist_args should be a list of keyward:value pairs.

      examples:
      1) ['ndim:5']
      2) ['ndim:10', 'base_distribution:uniform']
    """
    dist_name = args.dist
    kwargs = dict()
    if len(args.dist_args) > 0:
        for a in args.dist_args:
            k, v = a.split("=")
            if v.isdigit():
                v = int(v)
            kwargs.update({k: v})

    if dist_name in distributions.BaseDistribution.dist_names:
        dist = distributions.BaseDistribution.dists[dist_name](**kwargs)
    else:
        raise NotImplementedError(
            f"Got dist argument `{dist_name}`, not one of {distributions.BaseDistribution.dist_names}"
        )
    if args.atoms:
        dist = distributions.MixedDistribution(args.atoms, dist)
    return dist


def evaluate(args, model):

    all_runs = list()
    t_pvals = list()
    y_pvals = list()

    for _ in range(args.num_univariate_tests):
        uni_metrics = model.get_univariate_quant_metrics(dataset="test")
        all_runs.append(uni_metrics)
        t_pvals.append(uni_metrics["t_ks_pval"])
        y_pvals.append(uni_metrics["y_ks_pval"])

    summary = OrderedDict()
    summary.update(nll=model.best_val_loss)
    summary.update(avg_t_pval=sum(t_pvals) / args.num_univariate_tests)
    summary.update(avg_y_pval=sum(y_pvals) / args.num_univariate_tests)
    summary.update(min_t_pval=min(t_pvals))
    summary.update(min_y_pval=min(y_pvals))
    summary.update(q30_t_pval=np.percentile(t_pvals, 30))
    summary.update(q30_y_pval=np.percentile(y_pvals, 30))
    summary.update(q50_t_pval=np.percentile(t_pvals, 50))
    summary.update(q50_y_pval=np.percentile(y_pvals, 50))

    summary.update(ate_exact=model.ate().item())
    summary.update(ate_noisy=model.noisy_ate().item())

    return summary, all_runs


def main(args, save_args=True, log_=True):

    # create logger
    helpers.create(*args.saveroot.split("/"))
    logger = helpers.Logging(args.saveroot, "log.txt", log_)
    logger.info(args)

    # save args
    if save_args:
        with open(os.path.join(args.saveroot, "args.txt"), "w") as file:
            file.write(json.dumps(args.__dict__, indent=4))

    # dataset
    logger.info(f"getting data: {args.data}")
    ate, w, t, y = get_data(args)

    # comet logging
    if args.comet:
        exp = Experiment(project_name="causal-benchmark", auto_metric_logging=False)
        exp.add_tag(args.data)
    else:
        exp = None

    logger.info(f"ate: {ate}")

    # distribution of outcome (y)
    distribution = get_distribution(args)
    logger.info(distribution)

    # architecture
    mlp_params = MLPParams(
        n_hidden_layers=args.n_hidden_layers,
        dim_h=args.dim_h,
        activation=getattr(torch.nn, args.activation)(),
    )
    logger.info(mlp_params.__dict__)
    network_params = dict(
        mlp_params_w=mlp_params,
        mlp_params_t_w=mlp_params,
        mlp_params_y0_w=mlp_params,
        mlp_params_y1_w=mlp_params,
    )

    # training params
    training_params = TrainingParams(
        lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs
    )
    logger.info(training_params.__dict__)

    # initializing model
    w_transform = preprocess.Preprocess.preps[args.w_transform]
    y_transform = preprocess.Preprocess.preps[args.y_transform]
    outcome_min = 0 if args.y_transform == "Normalize" else None
    outcome_max = 1 if args.y_transform == "Normalize" else None

    model = TarNet(w, t, y,
                   training_params=training_params,
                   network_params=network_params,
                   binary_treatment=True, outcome_distribution=distribution,
                   outcome_min=outcome_min,
                   outcome_max=outcome_max,
                   train_prop=args.train_prop,
                   val_prop=args.val_prop,
                   test_prop=args.test_prop,
                   seed=args.seed,
                   early_stop=args.early_stop,
                   ignore_w=args.ignore_w,
                   grad_norm=args.grad_norm,
                   w_transform=w_transform, y_transform=y_transform,  # TODO set more args
                   savepath=os.path.join(args.saveroot, 'model.pt'),
                   test_size=args.test_size)
    

    # TODO GPU support
    if args.train:
        model.train(print_=logger.info, comet_exp=exp)

    # evaluation
    if args.eval:
        summary, all_runs = evaluate(args, model)
        logger.info(summary)
        with open(os.path.join(args.saveroot, "summary.txt"), "w") as file:
            file.write(json.dumps(summary, indent=4))
        with open(os.path.join(args.saveroot, "all_runs.txt"), "w") as file:
            file.write(json.dumps(all_runs))

        model.plot_ty_dists()

    return model


def get_args():
    parser = argparse.ArgumentParser(description="causal-gen")

    # dataset
    parser.add_argument("--data", type=str, default="lalonde")  # TODO: fix choices
    parser.add_argument(
        "--dataroot", type=str, default="datasets"
    )  # TODO: do we need it?
    parser.add_argument("--saveroot", type=str, default="save")
    parser.add_argument("--train", type=eval, default=True, choices=[True, False])
    parser.add_argument("--eval", type=eval, default=True, choices=[True, False])
    parser.add_argument('--overwrite_reload', type=str, default='',
                        help='secondary folder name of an experiment')  # TODO: for model loading

    # distribution of outcome (y)
    parser.add_argument('--dist', type=str, default='FactorialGaussian',
                        choices=distributions.BaseDistribution.dist_names)
    parser.add_argument("--dist_args", type=str, default=list(), nargs="+")
    parser.add_argument("--atoms", type=float, default=list(), nargs="+")

    # architecture
    parser.add_argument("--n_hidden_layers", type=int, default=1)
    parser.add_argument("--dim_h", type=int, default=64)
    parser.add_argument("--activation", type=str, default="ReLU")

    # training params
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--early_stop", type=eval, default=True, choices=[True, False])
    parser.add_argument("--ignore_w", type=eval, default=False, choices=[True, False])
    parser.add_argument("--grad_norm", type=float, default=float("inf"))
    parser.add_argument("--test_size", type=int)

    parser.add_argument('--w_transform', type=str, default='Standardize',
                        choices=preprocess.Preprocess.prep_names)
    parser.add_argument('--y_transform', type=str, default='Normalize',
                        choices=preprocess.Preprocess.prep_names)
    parser.add_argument("--train_prop", type=float, default=0.5)
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--test_prop", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--comet", type=eval, default=False, choices=[True, False])

    # evaluation
    parser.add_argument("--num_univariate_tests", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
