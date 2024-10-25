# -*- coding: utf-8 -*-


#from tkinter import E
import numpy as np
import pandas as pd
import csv
import glob
import time
import os
import os.path
from pathlib import Path
import sys
import utils
from utils import generalFunctions as genFunc
from collections import defaultdict, deque
import search
from search import construct_index as cni
import random
import pickle
from fuzzywuzzy import process
import argparse


# return tab_name:[column_id, column_name]
def extract_tab_map_from_indexres(index_result):
    tab_map_dic = {}
    for r in index_result:
        split_res = r.split(' || ')
        if split_res[0] in tab_map_dic:continue
        tab_map_dic[split_res[0]] = [split_res[-1], split_res[1]]
    return tab_map_dic
    
def find_similar_columns(query_columns, data_lake_columns, threshold=80):
    similar_columns = {}
    
    for query_col in query_columns:
        # Find the best matches for each query column in the data lake
        matches = process.extract(query_col, data_lake_columns, limit=1)
        
        # Filter matches by a similarity threshold
        filtered_matches = [col for col, score in matches if score >= threshold]

        if len(filtered_matches)>0:similar_columns[query_col] = filtered_matches[0]
    
    return similar_columns

                          
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process table data.")
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--k', type=int, default=10, 
                        help="Set the maximum count of tables using k")
    args = parser.parse_args()

    current_benchmark = args.benchmark
    current_mode = "estimate"
    benchmarkLoadStart = time.time()
    k = args.k

    QUERY_TABLE_PATH = os.path.abspath(f"../../{current_benchmark}_benchmark/injected_missing_query/")
    DATALAKE_PATH = os.path.abspath(f"../../{current_benchmark}_benchmark/datalake/")
    query_table_col_df = pd.read_csv(os.path.abspath(f'../../{current_benchmark}_benchmark/missing_query/missing_tab_row_col.csv'))

    dir_path = os.path.abspath(f'../')
    if not os.path.exists(os.path.join(dir_path,'test_stats')):
        os.makedirs(os.path.join(dir_path,'test_stats'), exist_ok=True)
    
    FINAL_RESULT_PICKLE_PATH = f"{dir_path}/test_stats/" + current_benchmark + "_benchmark_result_by_santos_"+current_mode+".pkl"

    num_save_index_path = f"{dir_path}/test_hashmap/{current_benchmark}_num_index.pkl"
    cate_save_index_path = f"{dir_path}/test_hashmap/{current_benchmark}_cate_index.pkl"


    num_lsh_index, num_hnsw_index, num_scaler, num_meta_data = pickle.load(open(num_save_index_path,'rb'))
    cate_lsh_index, cate_hnsw_index, cate_scaler, str_meta_data = pickle.load(open(cate_save_index_path,'rb'))

    benchmarkLoadEnd = time.time()
    difference = int(benchmarkLoadEnd - benchmarkLoadStart)
    print("Time taken to load benchmarks in seconds:",difference)
    print("-----------------------------------------\n")
    computation_start_time = time.time()

    all_query_time = {}
    all_cnt, cur_cnt = 0, 0
    total_queries = 1

    output_res_tab_mapped_infos = {}
    hit_cnt = 0
    table_semantic_graph = {}
    missing_col_search_time, semantic_construction_time = 0, 0
    # query_table_col_df = query_table_col_df.sample(1000)
    # query_table_col_df = query_table_col_df.sample(10, random_state=1)
    for idx, row in query_table_col_df.iterrows():
        table = os.path.join(QUERY_TABLE_PATH, row['table_path'])   
        table_name = row['table_path']
        
        query_long_index = f"{table_name} || {row['column_name']} || {row['column_id']}"
        if query_long_index in output_res_tab_mapped_infos:continue
        subject_index, subject_name = str(row['column_id']), row['column_name']

        print("Processing Table number:", total_queries, f'/ total {len(query_table_col_df)}')
        print("Table Name:", table_name, ', missing column ', row['column_name'])
        total_queries += 1

        current_query_time_start = time.time_ns()
        bagOfSemanticsFinal = []
        if not os.path.exists(table):continue
        input_table = pd.read_csv(table, encoding='latin1')
        stemmed_file_name = Path(table).stem

        # add the ensure of intent column
        a_start_time = time.time()
        if row['column_type']=='num':
            jaccard_query_result, topk_res, common_res = \
                cni.get_topk_jaccard_hnsw_cols(num_lsh_index, num_hnsw_index, num_scaler, num_meta_data, input_table, row['column_id'], intent_col_type='num', \
                    topk=min(50, len(num_meta_data)))
        elif row["column_type"]=='cate':
            jaccard_query_result, topk_res, common_res = \
                cni.get_topk_jaccard_hnsw_cols(cate_lsh_index, cate_hnsw_index, cate_scaler, str_meta_data, input_table, row['column_id'], intent_col_type='cate',\
                    topk=min(50, len(str_meta_data)))

            
        subject_mapped_result = extract_tab_map_from_indexres(common_res)
        if len(subject_mapped_result)==0:
            output_res_tab_mapped_infos[query_long_index] = {}
            print('not found for ', table_name, row['column_name'])
            continue
        a_end_time = time.time()
        missing_col_search_time +=  a_end_time - a_start_time
        print("missing column retrieve time ...", a_end_time - a_start_time, subject_mapped_result)

        current_index_for_subject_index = list(range(len(input_table.columns)))

        # Initialize dictionaries to track hit counts and candidate columns for each table
        tab_hit_cnt = {}
        tab_candidate_cols = {}
        query_sequence_cols = []

        # Collect all column indices for processing
        categorical_indices = []
        numerical_indices = []

        a_start_time = time.time()

        for related_col_idx in current_index_for_subject_index:
            related_col_idx = int(related_col_idx)
            query_related_col_name = input_table.columns[related_col_idx]
            query_sequence_cols.append(query_related_col_name)

            if genFunc.getColumnType(input_table.iloc[:, related_col_idx].tolist()) == 1:  # Check if the column type is categorical
                categorical_indices.append(related_col_idx)
            elif cni.is_numerical_col(input_table, input_table.columns[related_col_idx]):  # Check if the column type is numerical
                numerical_indices.append(related_col_idx)

        # Process categorical columns
        if categorical_indices:
            common_res_cate = cni.get_topk_hnsw_cols(
                cate_lsh_index, cate_hnsw_index, cate_scaler, str_meta_data, 
                input_table, categorical_indices, intent_col_type='cate', topk=min(50, len(str_meta_data))
            )

            for idx, common_res in zip(categorical_indices, common_res_cate):
                query_related_col_name = input_table.columns[idx]
                cur_mapped_dict = extract_tab_map_from_indexres(common_res)

                for tab in cur_mapped_dict.keys():
                    tab_hit_cnt[tab] = tab_hit_cnt.get(tab, 0) + 1
                for tab, col_info in cur_mapped_dict.items():
                    if tab not in tab_candidate_cols:
                        tab_candidate_cols[tab] = set()
                    tab_candidate_cols[tab].add((query_related_col_name, col_info[-1]))

        # Process numerical columns
        if numerical_indices:
            common_res_num = cni.get_topk_hnsw_cols(
                num_lsh_index, num_hnsw_index, num_scaler, num_meta_data, 
                input_table, numerical_indices, intent_col_type='num', topk=min(50, len(num_meta_data))
            )

            for idx, common_res in zip(numerical_indices, common_res_num):
                query_related_col_name = input_table.columns[idx]
                cur_mapped_dict = extract_tab_map_from_indexres(common_res)

                for tab in cur_mapped_dict.keys():
                    tab_hit_cnt[tab] = tab_hit_cnt.get(tab, 0) + 1
                for tab, col_info in cur_mapped_dict.items():
                    if tab not in tab_candidate_cols:
                        tab_candidate_cols[tab] = set()
                    tab_candidate_cols[tab].add((query_related_col_name, col_info[-1]))

        # Get the top k tables according to the hit counts in descending order
        topk_candidate_tables = [_ for _ in sorted(tab_hit_cnt.keys(), key=lambda x: tab_hit_cnt[x], reverse=True) if _ in subject_mapped_result.keys()][:k]
        a_end_time = time.time()
        semantic_construction_time += a_end_time - a_start_time
        print("candidate table retrieve time ...", a_end_time - a_start_time, subject_mapped_result)

        current_query_time_end = time.time_ns()
        all_query_time[query_long_index] = int(current_query_time_end - current_query_time_start)/10**9
        print("Time taken to process current table and print the results in seconds:", all_query_time[query_long_index])


        each_table_pos_res = {}
        for candidate_tab_name in topk_candidate_tables:
            with open(os.path.join(DATALAKE_PATH, candidate_tab_name), 'r') as file:
                candidate_tab_header = file.readline().strip().split(',')

            column_name_mapper = find_similar_columns(input_table.columns[current_index_for_subject_index], candidate_tab_header)
            for key, val in column_name_mapper.items():
                tab_candidate_cols[candidate_tab_name].add((key, val))  # Use add() instead of append()

            # print(candidate_tab_name, tab_candidate_cols[candidate_tab_name])
            query_sequence_cols, candidate_sequence_cols = [m[0] for m in tab_candidate_cols[candidate_tab_name]], [m[1] for m in tab_candidate_cols[candidate_tab_name]]
            if subject_name not in query_sequence_cols:
                query_sequence_cols.append(subject_name)
            if subject_mapped_result[candidate_tab_name][-1] not in candidate_sequence_cols:
                candidate_sequence_cols.append(subject_mapped_result[candidate_tab_name][-1])

            

            tmp_query_sequence_cols, tmp_candidate_sequence_cols = [], []
            for i in range(len(query_sequence_cols)-1,-1, -1):
                _ = query_sequence_cols[i]
                if _ in tmp_query_sequence_cols:continue
                tmp_query_sequence_cols.append(_)
                tmp_candidate_sequence_cols.append(candidate_sequence_cols[i])

            query_sequence_cols, candidate_sequence_cols = tmp_query_sequence_cols, tmp_candidate_sequence_cols
            each_table_pos_res[candidate_tab_name] = \
                int(subject_mapped_result[candidate_tab_name][0]), query_sequence_cols, candidate_sequence_cols

            # print('-------', candidate_tab_name, query_sequence_cols, candidate_sequence_cols)

        output_res_tab_mapped_infos[query_long_index] = each_table_pos_res

    computation_end_time = time.time()
    difference = int(computation_end_time - computation_start_time)
    print("Time taken to process all query tables and print the results in seconds:", difference)
print("Final Note: Find the missing columns in seconds:", missing_col_search_time)
print("Final Note: Time taken to process the semantic graph in seconds:", semantic_construction_time)
genFunc.saveDictionaryAsPickleFile(output_res_tab_mapped_infos, FINAL_RESULT_PICKLE_PATH)