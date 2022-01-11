# pyoptes
Python framework for optimization of epidemic testing strategies

## requirements
The code should work with the following python environment:
- python=3.7.6
- numba=0.48.0
- numpy=1.19.2
- scipy=1.4.1
- networkx=2.6.3
If the older numba=0.46.0 was used instead, an unresolved issue with data types arises.

## Getting started
- Try the script `test_target_function.py` to see the target function for optimizing the test budget allocation in action, to get an idea about how stochastic ("noisy") the target function evaluations are, and to see how several heuristically intuitive candidate inputs (=budget allocations) perform.
- Try the scripts `small_test.py` and `larger_test.py` to understand how the underlying components of the target function can be used: the synthetic generation of transmissions data (`pyoptes.networks.transmissions.scale_free`) and the simulation of an epidemiological model (`pyoptes.epidemiological_models.si_model_on_transmissions`).
- Look into the source codes of `pyoptes.networks.transmissions.scale_free` and `pyoptes.epidemiological_models.si_model_on_transmissions` how these work in detail.
