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
├── DataGenModel.py
	- workhorse class that takes in a dataset, model, guide, etc. and 
	  provides useful methods such as training, sampling, ATE calculation,
	  plotting diagnostics, quantitative diagnostics, etc.
├── examples.py		- file that includes many examples of how to use the code
├── plotting.py		- diagnostic plotting functions
└── utils.py		- various utility functions
```

## Testing
This project uses [pytest](https://docs.pytest.org/en/latest/). Before you push code, it is generally a good idea to run `pytest` to make sure it hasn't broken anything that is covered by the existing tests, especially if you are pushing a big commit. You can run `pytest -v` for a bit more verbose output. This will just run fast tests that only take a few seconds. You can run the slow tests that involve fully training models by including the `--runslow` flag.