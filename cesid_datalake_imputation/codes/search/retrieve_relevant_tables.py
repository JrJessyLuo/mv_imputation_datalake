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
import construct_index as cni
from construct_index import ColumnLSHEnsembleIndex
import random
import pickle
from fuzzywuzzy import process
import argparse

def find_similar_columns(query_columns, data_lake_columns, threshold=80):
    similar_columns = {}
    
    for query_col in query_columns:
        # Find the best matches for each query column in the data lake
        matches = process.extract(query_col, data_lake_columns, limit=1)
        
        # Filter matches by a similarity threshold
        filtered_matches = [col for col, score in matches if score >= threshold]

        if len(filtered_matches)>0:similar_columns[query_col] = filtered_matches[0]
    
    return similar_columns

# for numerical column of miss_idx, we can find the correlated groups
def find_community_for_query_miss(groups, input_table, miss_idx):
    grp_unique_ratio = {}
    for grp in groups:
        # Extract indices from group
        idx_grp = list(set([int(r) for item in grp for r in item.split('-')]))

        x_idx_grp = [miss_idx] + idx_grp

        # Calculate unique ratio
        unique_ratio = abs(
            len(input_table.iloc[:, x_idx_grp].drop_duplicates()) / len(input_table.iloc[:, idx_grp].drop_duplicates()) - 1
        )

        # Store the unique ratio for the group
        grp_unique_ratio[tuple(idx_grp)] = unique_ratio

    # Find and print the group with the minimum unique ratio
    min_unique_ratio_grp = min(grp_unique_ratio, key=grp_unique_ratio.get)

    return min_unique_ratio_grp

def find_community_for_query(groups, miss_idx):
    for grp in groups:
        # if miss_idx in grp:return grp
        # Extract indices from group
        idx_grp = list(set([int(r) for item in grp for r in item.split('-')]))

        if miss_idx in idx_grp:return tuple(idx_grp)

# each community is a semantic tree, each node index belong to a semantic tree
def getSemanticTreeCount(connect_pairs):
    # Build the graph as an adjacency list
    graph = defaultdict(list)
    for pair in connect_pairs:
        a, b = pair.split('-')
        graph[a].append(b)
        graph[b].append(a)

    # Function to find all connected components in the graph
    def find_communities(graph, pairs):
        visited = set()
        communities = []

        def bfs(start):
            community_nodes = []
            community_pairs = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                    community_nodes.append(node)
            # Add pairs that connect nodes in the community
            for a, b in pairs:
                if a in community_nodes and b in community_nodes:
                    community_pairs.append(f'{a}-{b}')
            return community_pairs

        for node in graph:
            if node not in visited:
                community = bfs(node)
                if community:
                    communities.append(community)
        
        return communities

    # Process pairs into a form suitable for the BFS function
    processed_pairs = [pair.split('-') for pair in connect_pairs]
    # Initialize a set to store unique indices
    unique_indices = set()

    # Iterate through processed pairs and add each index to the set
    for pair in processed_pairs:
        unique_indices.update(pair)

    # Convert the set to a sorted list if order matters
    sorted_indices = sorted(unique_indices, key=int)

    # Find all communities
    communities = find_communities(graph, processed_pairs)

    # Map each index to its corresponding community
    index_to_community = {}
    for community in communities:
        nodes_in_community = set()
        for pair in community:
            a, b = pair.split('-')
            nodes_in_community.update([a, b])
        for index in nodes_in_community:
            index_to_community[index] = community

    return communities, index_to_community, sorted_indices
  
"""
    extract relevant column pairs using external knowledge base
"""
def computeRelationSemantics(input_table, LABEL_DICT, FACT_DICT):
    # relation_bag_of_words = []
    total_cols = input_table.shape[1]
    #total_rows = input_table.shape[0]
    relation_dependencies = []
    entities_finding_relation = {}
    relation_dictionary = {}
    #compute relation semantics
    for i in range(0, total_cols-1):
            #print("i=",i)
        if genFunc.getColumnType(input_table.iloc[:, i].tolist()) == 1: 
            #the subject in rdf triple should be a text column
            for j in range(i+1, total_cols):
                semantic_dict_forward = {}
                semantic_dict_backward = {}
                #print("j=",j)
                column_pairs = input_table.iloc[:, [i, j]]
                column_pairs = (column_pairs.drop_duplicates()).dropna()
                unique_rows_in_pair = column_pairs.shape[0]
                total_kb_forward_hits = 0
                total_kb_backward_hits = 0
                #print(column_pairs)
                #assign relation semantic to each value pair of i and j
                for k in range(0, unique_rows_in_pair):
                    #print(k)
                    #extract subject and object
                    found_relation = 0
                    subject_value = genFunc.preprocessString(str(column_pairs.iloc[k,0]).lower())
                    object_value = genFunc.preprocessString(str(column_pairs.iloc[k,1]).lower())
                    is_sub_null = genFunc.checkIfNullString(subject_value)
                    is_obj_null = genFunc.checkIfNullString(object_value)
                    if is_sub_null != 0:
                        sub_entities = LABEL_DICT.get(subject_value, "None")
                        if sub_entities != "None":
                            if is_obj_null != 0:    
                                obj_entities = LABEL_DICT.get(object_value, "None")
                                if obj_entities != "None":
                                    #As both are not null, search for relation semantics
                                    for sub_entity in sub_entities:
                                        for obj_entity in obj_entities:
                                            #preparing key to search in the fact file
                                            entity_forward = sub_entity + "__" + obj_entity
                                            entity_backward = obj_entity + "__" + sub_entity
                                            relation_forward = FACT_DICT.get(entity_forward, "None")
                                            relation_backward = FACT_DICT.get(entity_backward, "None")
                                            if relation_forward != "None":
                                                found_relation = 1
                                                total_kb_forward_hits += 1
                                                #keep track of the entity finding relation. We will use this to speed up the column semantics search
                                                key = str(i)+"_"+subject_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {sub_entity}
                                                else:
                                                    entities_finding_relation[key].add(sub_entity)
                                                key  = str(j) + "_" + object_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {obj_entity}
                                                else:
                                                    entities_finding_relation[key].add(obj_entity)
                                                for s in relation_forward:
                                                    if s in semantic_dict_forward:
                                                        semantic_dict_forward[s] += 1 #relation semantics in forward direction
                                                    else:
                                                        semantic_dict_forward[s] = 1
                                            if relation_backward != "None":
                                                found_relation = 1
                                                total_kb_backward_hits += 1
                                                #keep track of the entity finding relation. We will use this for column semantics search
                                                key = str(i)+"_"+subject_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {sub_entity}
                                                else:
                                                    entities_finding_relation[key].add(sub_entity)
                                                
                                                key  = str(j)+"_"+object_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {obj_entity}
                                                else:
                                                    entities_finding_relation[key].add(obj_entity)
                                                
                                                for s in relation_backward:
                                                    if s in semantic_dict_backward:
                                                        semantic_dict_backward[s] += 1 #relation semantics in reverse direction
                                                    else:
                                                        semantic_dict_backward[s] = 1
                if len(semantic_dict_forward) > 0:
                    # relation_bag_of_words.append((max(semantic_dict_forward, key=semantic_dict_forward.get)+"-r", str(i)+"_"+str(j), max(semantic_dict_forward.values())/ total_kb_forward_hits))
                    relation_dependencies.append(str(i)+"-"+str(j))
                    relation_dictionary[str(i)+"-"+str(j)] = [(max(semantic_dict_forward, key=semantic_dict_forward.get), max(semantic_dict_forward.values())/ total_kb_forward_hits)]
                if len(semantic_dict_backward) >0:
                    # relation_bag_of_words.append((max(semantic_dict_backward, key=semantic_dict_backward.get)+"-r", str(j)+"_"+str(i), max(semantic_dict_backward.values())/ total_kb_backward_hits))
                    relation_dependencies.append(str(j)+"-"+str(i))
                    relation_dictionary[str(j)+"-"+str(i)] = [(max(semantic_dict_backward, key=semantic_dict_backward.get), max(semantic_dict_backward.values())/ total_kb_backward_hits)]

    return entities_finding_relation, relation_dependencies, relation_dictionary

# return tab_name:[column_id, column_name]
def extract_tab_map_from_indexres(index_result):
    tab_map_dic = {}
    for r in index_result:
        split_res = r.split(' || ')
        if split_res[0] in tab_map_dic:continue
        tab_map_dic[split_res[0]] = [split_res[-1], split_res[1]]
    return tab_map_dic

def find_mapped_index_pairs(mapped_indices):
    # Initialize dictionaries to store mappings
    graph1_to_graph2 = {}
    query_sequence_cols, candidate_sequence_cols = [], []

    # Iterate over the list of mapped index pairs
    for pair in mapped_indices:
        for mapping in pair:
            graph1_index, graph2_index = map(int, mapping.split('-'))
            graph1_to_graph2[graph1_index] = graph2_index
            if graph1_index not in query_sequence_cols:
                query_sequence_cols.append(graph1_index)
                candidate_sequence_cols.append(graph2_index)

    return graph1_to_graph2, query_sequence_cols, candidate_sequence_cols

def construct_semantic_graph(input_table, label_dict, fact_dict, which_mode):
    if which_mode == 1:
        entity_finding_relations, relation_dependencies, relation_dictionary = computeRelationSemantics(input_table, label_dict, fact_dict)
    else:
        relation_dependencies = []
    return relation_dependencies

                          
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process table data.")
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--kb_path', type=str, required=True, 
                        help="The yago knowledge base with labels and triples")
    parser.add_argument('--k', type=int, default=10, 
                        help="Set the maximum count of tables using k")
    parser.add_argument('--which_mode', type=int, default=1,
                        help="1-- Identification of relevant columns using knowledge base and dependency; 2-- Identification of relevant columns simpy using dependency")
    args = parser.parse_args()

    current_benchmark = args.benchmark
    
    which_mode = args.which_mode
    current_mode = "full"
    benchmarkLoadStart = time.time()

    YAGO_PATH = args.kb_path
    k = args.k

    QUERY_TABLE_PATH = os.path.abspath(f"../../{current_benchmark}_benchmark/injected_missing_query/")
    DATALAKE_PATH = os.path.abspath(f"../../{current_benchmark}_benchmark/datalake/")
    query_table_col_df = pd.read_csv(os.path.abspath(f'../../{current_benchmark}_benchmark/missing_query/missing_tab_row_col.csv'))
    
    LABEL_FILE_PATH = YAGO_PATH + "yago-wd-labels_dict.pickle" 
    FACT_FILE_PATH = YAGO_PATH + "yago-wd-facts_dict.pickle"

    dir_path = os.path.abspath(f'../')
    if not os.path.exists(os.path.join(dir_path,'test_stats')):
        os.makedirs(os.path.join(dir_path,'test_stats'), exist_ok=True)
    if not os.path.exists(os.path.join(dir_path,'groundtruth')):
        os.makedirs(os.path.join(dir_path,'groundtruth'), exist_ok=True)
    
    FINAL_RESULT_PICKLE_PATH = f"{dir_path}/test_stats/" + current_benchmark + "_benchmark_result_by_santos_"+current_mode+".pkl"
    
    #load pickle files to the dictionary variables
    FD_FILE_PATH = f"{dir_path}/groundtruth/" + current_benchmark + "_FD_filedict.pickle"

    fd_dict = genFunc.loadDictionaryFromPickleFile(FD_FILE_PATH)
    if which_mode == 1:
        label_dict = genFunc.loadDictionaryFromPickleFile(LABEL_FILE_PATH)
        fact_dict = genFunc.loadDictionaryFromPickleFile(FACT_FILE_PATH)
    else:
        label_dict = {}
        fact_dict = {}

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
        unique_values = input_table.nunique().max()
        rowCardinality = {}
        rowCardinalityTotal = 0
        bag_of_semantics_final = []
        col_id = 0
        stemmed_file_name = Path(table).stem


        if table_name not in fd_dict:
            print('missing')
        # continue

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

        if table_name not in table_semantic_graph:
            if len(input_table)>1000:
                relation_dependencies = construct_semantic_graph(input_table.sample(frac=0.1), label_dict, fact_dict, which_mode)
            else:
                relation_dependencies = construct_semantic_graph(input_table, label_dict, fact_dict, which_mode)
        else:
            relation_dependencies = table_semantic_graph[table_name] 
    
        current_relations = set()
        if table_name in fd_dict:
            current_relations = set(fd_dict[table_name])
        for item in relation_dependencies:    
            current_relations.add(item)
        if len(current_relations)==0:continue
        
        semantic_groups, node_idx_relations, current_column_index = getSemanticTreeCount(current_relations )

        semantic_group_related_col = {}
        for col in range(len(input_table.columns)):
            if str(col) not in current_column_index: 
                current_relations = find_community_for_query_miss(semantic_groups, input_table, col)
            else:
                current_relations =  find_community_for_query(semantic_groups, col)
            if current_relations not in semantic_group_related_col:
                semantic_group_related_col[current_relations] = []
            if col not in current_relations:
                semantic_group_related_col[current_relations].append(col)

        for key, val in semantic_group_related_col.items():
            if int(subject_index) in key or int(subject_index) in val:
                current_index_for_subject_index = list(key)
                current_index_for_subject_index.extend(val)

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

        # find the table mapping relations
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