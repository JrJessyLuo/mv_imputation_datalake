import os
import pandas as pd
import subprocess
import tqdm
from tqdm import tqdm
import argparse
import glob
import json
import pickle

def convert_tab_fds(tab_path):
    try:
        tab_df = pd.read_csv(tab_path)
        tab_name = tab_path.split('/')[-1]
        
        if len(tab_df) > 10000 or len(tab_df.columns)>10:
            
            if len(tab_df) > 1000:
                tab_df = tab_df.sample(n=1000)
            else:
                tab_df = tab_df.sample(n=len(tab_df))
            temp_file_path = os.path.abspath(f'tmp_files/{tab_name}')
            if not os.path.exists(os.path.abspath(f'tmp_files')):
                os.makedirs(os.path.abspath(f'tmp_files'), exist_ok=True)
            tab_df.to_csv(temp_file_path, index=False)
            
            # Run terminal command
            command = [
                'java', '-cp', 'metanome-cli-1.1.0.jar:FDep_improved-1.2-SNAPSHOT.jar',
                'de.metanome.cli.App',
                '--algorithm', 'de.metanome.algorithms.fdep.FdepAlgorithmHashValues',
                '--files', temp_file_path,
                '--file-key', 'Relational_Input'
            ]
            subprocess.run(command)
            
            # Remove the temporary file
            os.remove(temp_file_path)
        else:
            # Run terminal command
            command = [
                'java', '-cp', 'metanome-cli-1.1.0.jar:FDep_improved-1.2-SNAPSHOT.jar',
                'de.metanome.cli.App',
                '--algorithm', 'de.metanome.algorithms.fdep.FdepAlgorithmHashValues',
                '--files', tab_path,
                '--file-key', 'Relational_Input'
            ]
            subprocess.run(command)
    except:
        print(f'error with file {tab_path}')

def sortFDs():
    FDResults = glob.glob("./results/*_fds")
    fileDict = {}
    for file in FDResults:
        tableName = None
        with open(file, 'r', encoding = "utf-8") as rawInput:
           # with open(outputFile, 'w') as processed:
                # writer = csv.writer(processed)
                for line in rawInput:
                    fdDict = json.loads(line)
                    determinants = fdDict["determinant"]["columnIdentifiers"]
                    if len(determinants) == 1:
                        dependant = fdDict["dependant"]
                        tableName = dependant["tableIdentifier"]
                        rhs = dependant["columnIdentifier"]
                        lhs = determinants[0]["columnIdentifier"]
                        fd = lhs + "-" + rhs
                        if tableName not in fileDict:
                            fileDict[tableName] = [fd]
                        else:
                            fileDict[tableName].append(fd)   
    finalFileDict = {k: list(set(v)) for k, v in fileDict.items()} 
    return finalFileDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process fds.")
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    args = parser.parse_args()
    
    current_benchmark = args.benchmark
    file_list_path = os.path.abspath(f'../../{current_benchmark}_benchmark/injected_missing_query/*.csv')

    for tab_path in glob.glob(file_list_path):
        convert_tab_fds(tab_path)

    if not os.path.exists(os.path.abspath(f"../groundtruth/")):
        os.makedirs(os.path.abspath(f"../groundtruth/"), exist_ok=True)

    fileDict = sortFDs()
    outputFile=open(os.path.abspath(f"../groundtruth/{args.benchmark}_FD_filedict.pickle"), 'wb')
    pickle.dump(fileDict,outputFile, protocol=pickle.HIGHEST_PROTOCOL)
    outputFile.close()
        