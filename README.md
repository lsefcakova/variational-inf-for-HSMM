# variational-inf-for-HSMM

Adaptation of code from [pysvihmm ](https://github.com/dillonalaird/pysvihmm) by Nick Foti, Jason Xu and Dillon Laird to implement variational inference on Hidden Semi Markov Models using expansion formulation as described in Langrock, Zucchinni (2011).


The adapted code is converted to python 3 using `!2to3`

Contents
--------

### HMM Classes

`hmmbase.py` : Abstract base class for finite variational HMMs.

`hmmbatchcd.py` : Batch variational inference via coordinate ascent.


### Utilities

`util.py` : Miscellaneous files for HMM Classes and Test Classes.

### Testing 

`test_hmmbatchcd.py` : Testing function and plotting approximations

### Experiments and Results

`experiments.ipynb` : Unified results presentation

Authors
--------

Lenka Sefcakova 


