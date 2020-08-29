from models.distributions import distributions
from models import preprocess
from models.nonlinear import MLP, TrainingParams, MLPParams
from data.lalonde import load_lalonde

w, t, y = load_lalonde(rct=True)
# dist = distributions.MixedDistribution([0.0], distributions.LogLogistic())
dist = distributions.FactorialGaussian()
training_params = TrainingParams(lr=0.001, batch_size=64, num_epochs=100)
mlp_params_y_tw = MLPParams(n_hidden_layers=2, dim_h=1024)
early_stop = True
ignore_w = False

ATE = list()
BEST_VAL_LOSS = list()
PY = list()

for seed in range(20):
    mlp = MLP(w, t, y,
              training_params=training_params,
              mlp_params_y_tw=mlp_params_y_tw,
              binary_treatment=True, outcome_distribution=dist,
              outcome_min=0.0, outcome_max=1.0,
              train_prop=0.5,
              val_prop=0.1,
              test_prop=0.4,
              seed=seed,
              early_stop=early_stop,
              ignore_w=ignore_w,
              w_transform=preprocess.Standardize, y_transform=preprocess.Normalize)

    mlp.train()

    BEST_VAL_LOSS.append(mlp.best_val_loss)
    ATE.append(mlp.noisy_ate(n_y_per_w=1000))
    test = mlp.get_univariate_quant_metrics(dataset='test')
    PY.append( sum(mlp.get_univariate_quant_metrics(dataset='test')['y_ks_pval']  for  _ in range(32)) / 32 )