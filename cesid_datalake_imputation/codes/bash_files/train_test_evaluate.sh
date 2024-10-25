#!/bin/bash
# create a similar dataset for raw_benchmark in benchmark (this is for training)
raw_benchmark='test'
benchmark=${raw_benchmark}1
# how many missing values
missing_cnt=100

mkdir -p ../../${benchmark}_benchmark/query
cp -r ../../${raw_benchmark}_benchmark/injected_missing_query/* ../../${benchmark}_benchmark/query/
cp -r ../../${raw_benchmark}_benchmark/datalake/ ../../${benchmark}_benchmark/datalake

cd ../utils
python inject_missing_values.py --benchmark $benchmark --missing_cnt $missing_cnt

cd ../search
python retrieve_relevant_values.py --benchmark $benchmark --raw_benchmark $raw_benchmark

cd ../estimation
python row_acquisitor.py --benchmark $benchmark --raw_benchmark $raw_benchmark
python estimator.py --benchmark $benchmark

cd ../classification
python create_feats_labels.py --benchmark $raw_benchmark --train_benchmark $benchmark
python classifier.py --benchmark $raw_benchmark --train_benchmark $benchmark

cd ../evaluate
python total_evaluate.py --benchmark $raw_benchmark