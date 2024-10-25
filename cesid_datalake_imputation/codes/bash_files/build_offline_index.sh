#!/bin/bash

# benchmark name
benchmark='test'
# how many parallel processes
num_groups=2

# changing directory
cd ../search

# offline building the indexes seperately
for ((i=0; i<num_groups; i++))
do
    python construct_index.py --datatype cate --benchmark $benchmark --groupindex $i --num_groups $num_groups &
    python construct_index.py --datatype num --benchmark $benchmark --groupindex $i --num_groups $num_groups &
done

wait
# merge all the offline built indexes
python construct_index.py --datatype all --benchmark $benchmark --groupindex 0 --num_groups $num_groups
