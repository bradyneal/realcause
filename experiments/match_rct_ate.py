import torch

from models.distributions import distributions
from models import preprocess
from models.nonlinear import MLP, TrainingParams, MLPParams
from data.lalonde import load_lalonde

dataset = 2
if dataset == 1:
    w, t, y = load_lalonde()
    dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
    training_params = TrainingParams(lr=0.0005, batch_size=128, num_epochs=100, verbose=False)
    mlp_params_y_tw = MLPParams(n_hidden_layers=2, dim_h=256)
    early_stop = True
    ignore_w = False
elif dataset == 2:
    w, t, y = load_lalonde(rct=True)
    # dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
    dist = distributions.FactorialGaussian()
    training_params = TrainingParams(lr=0.001, batch_size=64, num_epochs=200)
    mlp_params_y_tw = MLPParams(n_hidden_layers=2, dim_h=1024)
    early_stop = True
    ignore_w = False
elif dataset == 3:
    w, t, y = load_lalonde(obs_version='cps1')
    dist = distributions.MixedDistribution([0.0, 25564.669921875 / y.max()], distributions.LogNormal())
    training_params = TrainingParams(lr=0.0005, batch_size=128, num_epochs=1000)
    mlp_params_y_tw = MLPParams(n_hidden_layers=3, dim_h=512, activation=torch.nn.LeakyReLU())
    early_stop = True
    ignore_w = False
else:
    raise(Exception('dataset {} not implemented'.format(dataset)))


mlp = MLP(w, t, y,
          training_params=training_params,
          mlp_params_y_tw=mlp_params_y_tw,
          binary_treatment=True, outcome_distribution=dist,
          outcome_min=0.0, outcome_max=1.0,
          train_prop=0.5,
          val_prop=0.1,
          test_prop=0.4,
          seed=1,
          early_stop=early_stop,
          ignore_w=ignore_w,
          w_transform=preprocess.Standardize, y_transform=preprocess.Normalize)

mlp.train()

naive = y[t == 1].mean() - y[t == 0].mean()
print('Naive ATE:', naive)
print('Noisy ATE est (using sample_y):', mlp.noisy_ate())
print('ATE est (using mean_y):', mlp.ate())
