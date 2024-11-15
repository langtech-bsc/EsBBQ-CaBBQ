#!/bin/bash

unset PYTHONPATH

# activate modules
module load mkl intel impi hdf5 python/3.12.1-gcc sqlite3/3.45.2-gcc

# activate the virtual env
export PYTHONPATH=/gpfs/projects/bsc88/text/bias/bbq/venv/lib/python3.12/site-packages
source /gpfs/projects/bsc88/text/bias/bbq/venv/bin/activate
