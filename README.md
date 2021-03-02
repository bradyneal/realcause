# causal-benchmark
Realistic benchmark for different causal inference methods. The realism comes from fitting generative models to data with an assumed causal structure. 

## Installation
1. `conda env create -f environment.yml`
2. `conda activate gen`

This installs all the dependencies in a conda environment and activates the environment. To deactivate the environment, run `conda deactivate`. The environment can be reactivated at any time with `conda activate gen`.

If step 1 above fails, try the following instead: `conda env create -f environment_exact.yml`.


## Project Structure
```
├── data/		- folder for data generators and real datasets
├── models/		- folder for Pyro/PyTorch models and guides
├── plots/		- folder where (gitignored) plots are automatically saved
├── tests/		- folder for test suite
├── benchmarking.py	- benchmark estimators on data generators
├── DataGenModel.py
	- workhorse class that takes in a dataset, model, guide, etc. and 
	  provides useful methods such as training, sampling, ATE calculation,
	  plotting diagnostics, quantitative diagnostics, etc.
├── examples.py		- file that includes many examples of how to use the code
├── plotting.py		- diagnostic plotting functions
├── slides.pdf		- project slides from presentation Brady gave for a class
└── utils.py		- various utility functions
```

## Testing
This project uses [pytest](https://docs.pytest.org/en/latest/). Before you push code, it is generally a good idea to run `pytest` to make sure it hasn't broken anything that is covered by the existing tests, especially if you are pushing a big commit. You can run `pytest -v` for a bit more verbose output. This will just run fast tests that only take a few seconds. You can run the slow tests that involve fully training models by including the `--runslow` flag. You can run the plotting tests by including the `--runplot` flag.

## Installing whynot_estimators
The regular `whynot` package comes with [3 simple estimators](https://github.com/zykls/whynot/blob/master/whynot/algorithms/causal_suite.py#L47): OLS, propensity score matching, and propensity score weighted OLS. These may be sufficient for your purposes. If not, read further.

To use more, you must install them in the `whynot_estimators` package. The `whynot_estimators` package should already have been pip installed from environment.yml. To use the additional 4 estimators [here](https://github.com/zykls/whynot/blob/master/whynot/algorithms/causal_suite.py#L53), you must install them as follows:
```
python -m whynot_estimators install ip_weighting
python -m whynot_estimators install matching
python -m whynot_estimators install causal_forest
python -m whynot_estimators install tmle
```

Depending on your operating system, you may have issues with these installs because conda doesn't always play well with r.

For example, on MacOS, I had trouble installing the `Matching` R package and the `dbarts` R package that is used in the `tmle` R package. I just followed [this answer StackOverflow answer](https://stackoverflow.com/a/55875539) and ran the following r code inside the `gen` conda env to fix the issue:

```r
(gen) $ r
> Sys.setenv(CONDA_BUILD_SYSROOT="/")
> install.packages("Matching")
> install.packages("tmle")
> install.packages("dbarts")
```


## Training causal generative models


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

* `--data`

TODO #brady


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
 