## Repository Organization
- codes
  - bash_files: Contains commands to execute the project using the example data lake (test).
  - classification: Code for the classification module.
  - estimation: Code for the estimation module
  - search: Codes for the search module.
  - fd_tools: Code for detecting relevant columns based on dependency. Download the files `FDep_improved-1.2-SNAPSHOT.jar` and `metanome-cli-1.1.0.jar` (refer to [link](https://github.com/sekruse/metanome-cli)), and then place them into it.
  - evaluate: Evaluation results of the imputed values.
  - utils: Basic codes.
- data
  - entitable: The test missing values in the data lake EntiTables.
  - webtable: The test missing values in the data lake WebTable.
  - opendata: The test missing values in the data lake OpenData.
- requirements.txt: List of necessary packages to run the project.
  
## Benchmark
- EntiTables: The benchmark is available at [this link](https://github.com/iai-group/cikm2019-table).
- WebTable and OpenData: The two benchmarks are available at [this link](https://github.com/RLGen/LakeBench).


## Reproducibility
1. Install necessary packages using the following commands. Additionally, install [Sherlock](https://github.com/mitmedialab/sherlock-project/tree/f194013e795d90bc63553a2c47bb33f56c1c7b53).
   ```
   conda create --name cesid python=3.9
   conda activate cesid
   pip install -r requirements.txt
   ```
NOTE: The following steps are based on the example data lake named **test**. If you are using a different data lake (e.g., xx), replace all instances of test with xx, including commands in the bash_files directory.

2. Set the environment
  ```
  export PYTHONPATH=<path_to_codes_directory>
  export yagopath=<path_to_yago_pickle>
 ```
- PYTHONPATH should point to the directory path of **codes**.
- yagopath should point to the directory path of **yago_pickle** (For more details on setting this up, please refer to [this link](https://github.com/northeastern-datalab/santos)).

3. Prepare the Data Lake
  - If you are creating a new data lake (e.g., `test`), you should first follow these steps:
```
cd cesid_datalake_imputation/codes
mkdir test_benchmark
mkdir test_benchmark/query
mkdir test_benchmark/datalake
```
Place the chosen **incomplete tables** in `test_benchmark/query` and the **data lake tables** in `test_benchmark/datalake`.
And inject the missing values into the data lake: 
```
cd utils
python inject_missing_values.py --benchmark test --missing_cnt 100
```
  - Using our Provided Data Lake (e.g., `OpenData`), follow these steps:
```
cd cesid_datalake_imputation/codes
mkdir opendata_benchmark
mkdir opendata_benchmark/query
mkdir opendata_benchmark/datalake
```
Place the chosen incomplete tables in `opendata_benchmark/query` and the data lake tables in `opendata_benchmark/datalake`.
Place the missing value in the benchmark:
```
mkdir opendata_benchmark/missing_query
cp ../data/entitables/missing_tab_row_col.csv opendata_benchmark/missing_query/
```
  
4. Run the search module
   
   Remember to update the benchmark as needed:
   ```
   cd cesid_datalake_imputation/codes/bash_files
   bash build_offline_index.sh
   bash search_identification.sh
   ```
5. Run the estimation module
   
   Remember to update the benchmark as needed:
   ```
   cd cesid_datalake_imputation/codes/bash_files
   bash estimation_output.sh
   ```
6. Run the classification module
   
   Remember to update the benchmark as needed:
   ```
   cd cesid_datalake_imputation/codes/bash_files
   bash train_test_evaluate.sh
   ```
