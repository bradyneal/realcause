# causal-benchmark
Realistic benchmark for different causal inference methods. The realism comes from fitting generative models to data with an assumed causal structure. 

## Installation
Once you've created a virtual environment (e.g. with conda, virtualenv, etc.) install the required packages:

```
pip install -r requirements.txt
```

## Do your own analysis on our causal-predictive metric dataset

We trained a total of 1568 different estimators.
We recorded all of the predictive metrics that sklearn provides (e.g. RMSE, MAE, precision, recall, etc.) and many different causal metrics that RealCause provides (e.g. ATE bias, ATE RMSE, PEHE, etc.).
Taking all of these metrics plus estimator specification (meta-estimator, outcome model, and propensity score model) yields a total of 77 columns.
Cells are "nan" where that cell doesn't make sense (e.g. the propensity score model cell for a standardization estimator, a regression metric for an IPW estimator, a classification metric for a standardization estimator, etc.).

We provide this dataset in [causal-predictive-analysis.csv](https://github.com/bradyneal/causal-benchmark/blob/master/causal-predictive-analysis.csv).
We did one analysis on this dataset in Section 6 of our paper (in [experiments/uai_analysis.py](https://github.com/bradyneal/causal-benchmark/blob/master/experiments/uai_analysis.py)).
However, there are many more possible analyses that can be run on it.
For example, one might want to fit machine learning models to predict causal metrics from predictive metrics and use something like [SHAP](https://github.com/slundberg/shap) to interpret the associations these models find.
To get started, simply load the dataset from [causal-predictive-analysis.csv](https://github.com/bradyneal/causal-benchmark/blob/master/causal-predictive-analysis.csv).
Example loading:

```
import pandas as pd

df = pd.read_csv('causal-predictive-analysis.csv')
```

## Loading RealCause pre-computed datasets

You can load any of the realistic RealCause datasets (trained on LaLonde PSID, LaLonde CPS, and Twins) from `realcause_datasets/` using `pandas.read_csv()` or by using our `load_realcause_dataset()` function in `loading.py`.
We provide 100 different samples of each dataset.
These samples are generated in `make_datasets.py`.


Example usage to load sample 69 of the LaLonde PSID dataset:

```
from loading import load_realcause_dataset

df = load_realcause_dataset('lalonde_psid', 69)
```

Valid value for the dataset argument: 'lalonde\_psid', 'lalonde\_cps', and 'twins'.
Valid values for the sample argument: 0-99.
If the sample argument is not given, it defaults to 0.

Example usage to load sample 0 of the Twins dataset without giving the sample argument:

```
from loading import load_realcause_dataset

df = load_realcause_dataset('twins')
```


## Loading RealCause pre-trained generative models
Loading the pre-trained models can be done using the function `load_from_folder(DATASET)` from the script `loading.py`, where `DATASET` can be one of:

 - `lalonde_cps1`
 - `lalonde_psid1`
 - `LBIDD_exp`
 - `LBIDD_linear`
 - `LBIDD_log`
 - `LBIDD_quadratic`
 - `ihdp`
 - `twins`
 
For example, this is a script to load the model trained on the LaLonde CPS dataset:

```
from loading import load_from_folder
model, args = load_from_folder("lalonde_cps1")
```

## Using RealCause generative models

### Sampling

To see most of the methods you can use with these generative models, see the [BaseGenModel class](https://github.com/bradyneal/causal-benchmark/blob/master/models/base.py#L56).
After you've loaded a generative model `model`, you can sample from it as follows:

```
w, t, y = model.sample()
```

We show how to use the knobs below.
See further documentation for the sample method in [its docstring](https://github.com/bradyneal/causal-benchmark/blob/master/models/base.py#L322).

### Using knobs

We currently provide three knobs as parameters to the `sample()` method:

* `overlap`
	- If 1, leave treatment untouched.
	- If 0, push p(T = 1 | w) to 0 for all w where p(T = 1 | w) < 0.5 and push p(T = 1 | w) to 1 for all w where p(T = 1 | w) >= 0.5.
	- If 0 < overlap < 1, do a linear interpolation of the above.
* `causal_effect_scale`: scale of the causal effect (size of ATE)
* `deg_hetero`: degree of heterogeneity (between 0 and 1). When `deg_hetero=1`, y<sub>1</sub> and y<sub>0</sub> remain unchanged. When `deg_hetero=0`,
            y<sub>1</sub> - y<sub>0</sub> is the same for all individuals.

## Training RealCause generative models


<!--### Code structure

Our deep generative model assumes the following factorization

```
p(w, t, y) = p(w)p(t|w)p(y|t,w)
```

so that a random sample of the tuple (w,t,y) can be drawn from the joint distribution via ancestral sampling. 

We let p(w) be the empirical distribution of the training set, and parameterize p(t|w) and p(y|t,w) using
neural networks (or other conditional generative models such as Gaussian processes). 
The model is defined in `models/tarnet.py`. The neural networks defined there will output the parameters for 
the distribution classes defined in `models/distributions` and compute the negative log likelihood as the loss function.


### Training loop-->

The main training script is `train_generator.py`, which will run one experiment for
a set of hyperparameter (hparam) configuration. The hparams include `--batch_size`, `--num_epochs`, `--lr`, etc. 
Here's an example command line:

```bash
python train_generator.py --data "lalonde" --dataroot [path-to-ur-data-folder] --saveroot [where-to-save-stuff] \
    --dist "FactorialGaussian" --n_hidden_layers 1 --dim_h 128 --w_transform "Standardize" --y_transform "Normalize"
```

* `--data` <br>
	This argument specifies the dataset. Options:
	- "lalonde" or "lalonde_psid" - LaLonde PSID dataset
	- "lalonde_cps" - LaLonde CPS dataset
	- "lalonde_rct" - LaLonde RCT dataset
	- "twins" - Twins dataset
	- "ihdp" - IHDP dataset
	- "lbidd\_\<link\>\_\<n\>" - LBIDD dataset with link function \<link\> and number of samples \<n\> <br>
		Valid \<link\> options: linear, quadratic, cubic, exp, and log <br>
		Valid \<n\> options: 1k, 2.5k, 5k, 10k, 25k, and 50k <br>
		Example: "lbidd\_cubic\_10k" yields an LBIDD dataset wth a cubic link function and 10k samples


* `--x_transform` <br>
This argument will tell the model to preprocess the covariate (W) or the outcome (Y) via "Standarization" 
(so that after the transformation the training data is centered and has unit variance) or via "Normalization"
(so that after the transformation the training data will range from 0 to 1); the preprocessor uses training set's
statistics. 
<br>
If "Normalize" is applied to the outcome (Y), we further clamp the sample outcome value at 0 and 1, so that we do not
generate samples outside of the min-max range of the training set.

* `--dist` <br>
This argument determines which distribution to be used for the outcome variable 
(we assume binary / Bernoulli treatment for this training script). To see a list of available distributions, run
```bash
python -c "from models.distributions import distributions; print(distributions.BaseDistribution.dist_names)"
['Bernoulli', 'Exponential', 'FactorialGaussian', 'LogLogistic', 'LogNormal', 'SigmoidFlow', 'MixedDistribution']
```

In most of our experiments, we use a more flexible family of distributions called normalizing flows; 
more specifically we use the [Sigmoidal Flow](https://arxiv.org/abs/1804.00779), which is a universal density model 
suitable for black-box Auto-ML. It is similar to mixture of distributions (like Gaussian mixture model), which 
has the ability to model multimodal distributions. 

In some cases (such as the Lalonde dataset), there might be discrete "atoms" presented in the dataset, which means the 
outcome variable is mixed-continuous-discrete-valued. We then have a special argument `--atoms` to model the probability that 
the outcome takes certain discrete values (given W and T). 

Concretely,

```bash
python train_generator.py --data "lalonde" ... \
    --dist "SigmoidFlow" \
    --dist_args "ndim=10" "base_distribution=gaussian" \ 
    --atoms 0 0.2
```

Note that the atom values (and distribution arguments) are separaeted by white space. 
For Sigmoidal Flow, there is an additional option for distribution arguments, whose
key (e.g. what base distribution to use for the flow) and value (e.g. gaussian) are separated by `=`. 
Valid choices for base distributions are `uniform` or `gaussian` (or `normal`). 
The `ndim` argument correspond to the "number of hidden units" of the sigmoid flow 
(think of it as an invertible 2-layer MLP). It is analogous to the number of mixture components of 
a mixture of Gaussian model.


## Training loop 
We also provide a convenient hyperparameter search script called `train_generator_loop.py`. 
It will load the `HP` object from `hparams.py`, and create a list of hparams by taking the Cartesian product of 
of the elements of `HP`. It will then spawn multiple threads to run the experiments in parallel. 

Here's an example using the default `hparams.py` (remember to change the `--dataroot`!):

```bash
python train_generator_loop.py --exp_name "test_flow_and_atoms" --num_workers=2
```

Note that `--saveroot` will be ignored by this training loop, since it will create an experiment folder and then create 
multiple hparam folders inside; and `--saveroot` will then be set to these folders. In the above example, there will be
4 of them:


```text
├── test_flow_and_atoms
│   ├── dist_argsndim=5+base_distribution=uniform-atoms
│   └── dist_argsndim=5+base_distribution=uniform-atoms0.0
│   └── dist_argsndim=10+base_distribution=normal-atoms
│   └── dist_argsndim=10+base_distribution=normal-atoms0.0
```

Once an experiment (for a single hparam setting) is finished, you should see 5 files in the hparam folder (saveroot).

```text
├── test_flow_and_atoms
│   ├── dist_argsndim=5+base_distribution=uniform-atoms
│   │   ├── args.txt
│   │   ├── log.txt
│   │   ├── model.pt
│   │   ├── all_runs.txt
│   │   ├── summary.txt
```

* args.txt: arguments of the experiment
* log.txt: all the "prints" of the experiment are redirected to this log file
* model.py: early-stopped model checkpoint
* all_runs.txt: 
univariate evaluation metrics (i.e. p values & nll; by default there will be `--num_univariate_tests=100` entries)
* summary.txt: summary statistics of all_runs.txt (mean and some quantiles).

## Re-running our causal estimator experiments

To re-run the causal estimator benchmarking in our paper, run [experiments/uai_experiments.py](https://github.com/bradyneal/causal-benchmark/blob/master/experiments/uai_experiments.py). To re-run our correlation analysis between causal and predictive metrics, run [experiments/uai_analysis.py](https://github.com/bradyneal/causal-benchmark/blob/master/experiments/uai_analysis.py).
 
