#!/bin/bash
# set the benchmark name
benchmark='test'

cd ../estimation
python table_acquisitor.py --benchmark $benchmark

python row_acquisitor.py --benchmark $benchmark --raw_benchmark $benchmark

python estimator.py --benchmark $benchmark