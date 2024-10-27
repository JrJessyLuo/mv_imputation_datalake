#!/bin/bash

benchmark='test'

# # detect the dependency for the table 
cd ../fd_tools
python generate_fds.py --benchmark $benchmark

# # retrieve relevant tables
cd ../search
python retrieve_relevant_tables.py --benchmark $benchmark --kb_path  $yagopath --which_mode 1

# retrieve relevant values
python retrieve_relevant_values.py --benchmark $benchmark --raw_benchmark $benchmark
