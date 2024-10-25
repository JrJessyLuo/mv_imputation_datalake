



## how to use our search module
cd codes/bash_files
bash build_offline_index.sh
bash online_table_identification.sh


## pip install tree_influence, sherlock
conda create --name cesid python=3.9
conda activate cesid
pip install -r requirements.txt

Install sherlock following https://github.com/mitmedialab/sherlock-project/tree/f194013e795d90bc63553a2c47bb33f56c1c7b53

## all the things that you should prepare
-- query tables, and data lake files; 
-- must include xx_benchmark, xx_benchmark/query, xx_benchmark/datalake; 
-- Using our benchmark, must include xx_benchmark/missing_query/missing_tab_row_col.csv

-- how to inject missing values
python inject_missing_values.py --benchmark test --missing_cnt 100

what I should put into the zenodo --yago
export PYTHONPATH=
export yagopath=
datasketch
cd bash_files

