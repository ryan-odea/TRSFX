# TRSFX - Time Resolved Serial Femtosecond X-ray Crystallography
[![PyPI version](https://badge.fury.io/py/TRSFX.svg)](https://pypi.org/project/TRSFX)
[![Downloads](https://static.pepy.tech/badge/TRSFX)](https://pepy.tech/project/TRSFX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![versions](https://img.shields.io/pypi/pyversions/TRSFX.svg)
[![Documentation Status](https://readthedocs.org/projects/TRSFX/badge/?version=latest)](https://TRSFX.readthedocs.io)


This package consists of many small tools for interacting between much larger packages in the field of crystallography, aimed at getting labs up and running at beamtimes much faster through a single import, rather than building out an entire new conda environment each time you travel. There are some dependencies of system packages, which would have to be installed via an administrator or depend on user modification of underlying code.

You can install the base package from pypi with
```bash
pip install TRSFX
```

## Package Use
While many of the functions can also be accessed through the python API, the majortiy of users might benefit from interacting with the CLI counterparts. Most of the functions are 'bundled' into submodules as to not overwhelm the user immediately. The current bundles are:

1. `sfx.compare` - Tools to form comparisons between HKL and MTZ files
2. `sfx.manip` - Tools to manipulate MTZ files

Calling any of these in the command line will bring up the helpfile and showcase functions available within them.

## Better documentation (and Examples!)
More complete documentation (and examples) are available through our [readthedocs website](https://trsfx.readthedocs.io/en/latest/). We invite you to take a look at our vignettes.
