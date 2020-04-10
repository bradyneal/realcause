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
This project uses [pytest](https://docs.pytest.org/en/latest/). Before you push code, it is generally a good idea to run `pytest` to make sure it hasn't broken anything that is covered by the existing tests, especially if you are pushing a big commit. You can run `pytest -v` for a bit more verbose output. This will just run fast tests that only take a few seconds. You can run the slow tests that involve fully training models by including the `--runslow` flag.

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