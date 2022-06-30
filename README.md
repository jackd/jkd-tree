# JKD Tree

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Jax](https://github.com/google/jax) port of [scikit-learn kdtrees](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neighbors).

## Project Status

Preliminary work on heaps (required for some KDTree operations) shows jax jitting performs poorly for algorithms involving many loops compared to numba, so this work is being paused until performance improves. See [numba-neighbors](https://github.com/jackd/numba-neighbors) for a numba implementation.

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
