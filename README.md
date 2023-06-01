# variational-inf-for-HSMM

Adaptation of code from [pysvihmm ](https://github.com/dillonalaird/pysvihmm) by Nick Foti, Jason Xu and Dillon Laird to implement variational inference on Hidden Semi Markov Models using expansion formulation as described in Langrock, Zucchinni (2011).

The adapted code is converted to python 3 using `2to3`.

Run `python setup.py build_ext --inplace` to build external Cython modules.

Contents
--------

### HMM Classes

Original classes implementing VI/SVI on a HMM data (in Python 3, with slight modifications from the original code)

`hmmbase.py` : Abstract base class for finite variational HMMs.

`hmmbatchcd.py` : Batch variational inference via coordinate ascent.

`hmmbatchsgd.py` : Batch stochastic variational inference via coordinate ascent.

`hmmsgd_metaobs.py` : SVI with batches of meta-observations. A meta-observation is a group of consecutive observations. We then form minibatches from these. The natural gradient for the global variables is computed for all observations in a meta-observation, and then those are averaged over all meta-observations in the minibatch.

### HSMM Classes

New classes implementing VI/SVI on a HSMM data (in Python 3) utilizing matrix expansion method mentioned above. The files correspond to their equivalents in HMM setting.

`hsmmbase.py` 

`hsmmbatchcd.py` 

`hsmmbatchsgd.py` 

`hsmmsgd_metaobs.py` 

### Utilities

`util.py` : Miscellaneous files for HMM Classes and Test Classes.

`matrix_expansion.py` : Expansion Matrix function implementation

`generate_data.py` : Synthetic data generator 

### Testing 

Testing function and plotting approximations.

`test_hmmbatchcd.py` : HMM VI

`test_hmmbatchsgd.py` : HMM SVI

`test_hsmmbatchcd.py` : HSMM VI

`test_hsmmbatchsgd.py` : HSMM SVI

### Experiments and Results

Results presented in Variational Inference for Hidden Semi-Markov Models Thesis project DSM 2023 at BSE.

`hmm_plots.ipynb`: Unified HMM results presentation.

`hsmm_plots.ipynb` : Unified HSMM results presentation.

Authors
--------

Lenka Sefcakova 

David Vallmanya
