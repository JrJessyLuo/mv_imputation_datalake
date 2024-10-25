import os
import pandas as pd
import time
import bz2
import pickle
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import difflib
import sys
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import dask.dataframe as dd
from collections import defaultdict
import utils
from utils import match_row as mr


def find_rows_with_most_overlap_dask(dataframe, value_set):
    value_set = set(value_set)

    def count_overlap(row):
        return len(set(row).intersection(value_set))

    ddf = dd.from_pandas(dataframe, npartitions=10)  # Adjust npartitions based on your system
    ddf['overlap_count'] = ddf.apply(count_overlap, axis=1, meta=('overlap_count', 'int64'))
    
    sorted_ddf = ddf.nlargest(n=100, columns='overlap_count')  # Adjust n as needed

    return sorted_ddf.compute()


def sort_aggregated_scores(trans_output_res):
    # Initialize an empty dictionary to store the sorted results
    sorted_output_res = {}

    # Iterate through the aggregated results
    for key, candidate_scores in trans_output_res.items():
        # Sort the candidate values by their scores in descending order
        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)

        # Extract just the candidate names (age group) in the sorted order
        sorted_output_res[key] = [candidate for candidate, _ in sorted_candidates]

    return sorted_output_res

def tuple_serializer(tuple_t, target_att=None): 
    """Returns string of serialized tuple."""
    if target_att is not None:
        all_cols = [k for k in tuple_t.keys() if k != target_att]
    else:
        all_cols = tuple_t.keys()
    arr = [(col.replace('\r\n', ' ').replace('\n',  ' '), str(tuple_t[col]).replace('\r\n', ' ').replace('\n',  ' ')) for col in all_cols]
    serialization = (' ; ').join([f'{pair[0]} : {pair[1]}' for pair in arr])
    if target_att is not None:
        serialization = f'{serialization} ; {target_att} : '
    serialized_tuple = f'[ {serialization} ]'
    return serialized_tuple

def preprocess_and_embed_rows(table, query_cols, target_att=None):
    """Preprocess and compute embeddings for the rows in a table."""
    # Drop duplicates and keep track of the original indices
    simplified_table = table.iloc[:, query_cols].drop_duplicates()
    serialized_rows = [tuple_serializer(row, target_att) for _, row in simplified_table.iterrows()]
    embeddings = model.encode(serialized_rows)
    return np.array(embeddings), simplified_table.index.tolist()

def get_single_column(query_col, real_col_lst):
    if query_col in real_col_lst:return query_col
    else:
        try:
            return difflib.get_close_matches(query_col, real_col_lst, n=1)[0]
        except:
            return 'error_xx1'


def get_columns(query_col_lst, real_col_lst):
    output_column_lst = []
    for q_col in query_col_lst:
        q_match_col = get_single_column(q_col, real_col_lst)
        if q_match_col != 'error_xx1':
            output_column_lst.append(q_match_col)
    return output_column_lst

def aggregate_scores(output_res):
    trans_output_res = defaultdict(dict)

    for query_id, search_results in output_res.items():
        value_score_dict = defaultdict(float)

        # Collect all scores for each candidate value and sum them
        for candidate_value, candidate_table, score in search_results:
            value_score_dict[candidate_value] += score

        # Store the aggregated sum of scores
        for candidate_value, total_score in value_score_dict.items():
            trans_output_res[query_id][candidate_value] = total_score

    return trans_output_res

def process_query_table(query_table_col_df, benchmark_path, candidate_map_res):
    output_res = {}
    datalake_tabs = {}
    datalake_tabs_header = {}
    potential_tabs = {key for related_infos in candidate_map_res.values() for key in related_infos.keys()}
    
    for pt_tab in tqdm(potential_tabs, total=len(potential_tabs)):
        # if 'CAN_CSV0000000000000910' not in pt_tab:continue
        pt_tab_path = os.path.join(benchmark_path, 'datalake', pt_tab)
        datalake_tabs[pt_tab] = pd.read_csv(pt_tab_path, encoding='latin1')
        datalake_tabs_header[pt_tab] = datalake_tabs[pt_tab].columns.tolist()

    print(f'{len(datalake_tabs)} unionable tables loading done...')

    query_tab_cache = {}
    for query_tab in query_table_col_df['table_path'].unique().tolist():
        # if query_tab!='CAN_CSV0000000000014395.csv':continue
        table_path = os.path.join(benchmark_path, "injected_missing_query", query_tab)
        if not os.path.exists(table_path):
            continue
        input_table = pd.read_csv(table_path, encoding='latin1')
        query_tab_cache[query_tab] = input_table
    print(f'{len(query_tab_cache)} query tables loading done...')

    all_search_time, categorical_search_time, numerical_search_time = 0, 0, 0
    all_cnt = 0

    table_column_dict = {}

    for idx, row in tqdm(query_table_col_df.iterrows(), total=len(query_table_col_df)):
        table_name = row['table_path']
        # if table_name!='CAN_CSV0000000000014395.csv':continue
        input_table = query_tab_cache[table_name]

        query_long_index = f"{table_name} || {row['column_name']} || {row['column_id']}"
        query_index = row['row_id']
        query_id = f"{table_name} || {row['column_name']} || {row['column_id']} || {row['row_id']}"

        # if query_long_index !='CAN_CSV0000000000014395.csv || UOM_ID || 8':continue

        if query_long_index not in candidate_map_res:
            continue

        missing_attr = row['column_id']
        missing_type = row['column_type']

        # Set the range for each column
        if f'{table_name}_{missing_attr}' not in table_column_dict:
            if missing_type == 'cate':
                table_column_dict[f'{table_name}_{missing_attr}'] = \
                    input_table.iloc[:, missing_attr].unique().tolist()
            elif missing_type == 'num':
                max_, min_ = input_table.iloc[:, missing_attr].max(), input_table.iloc[:, missing_attr].min()
                table_column_dict[f'{table_name}_{missing_attr}'] = \
                    [max_, min_]

        start_time = time.time()

        for candidate_name, query_candidate_map_cols in candidate_map_res[query_long_index].items():
            if len(query_candidate_map_cols) == 0:
                continue
            
            map_id, query_cols, candidate_cols = query_candidate_map_cols
            if len(query_cols) == 0:
                continue

            query_cols = get_columns(query_cols, input_table.columns.tolist())
            missing_column = get_single_column(row['column_name'], input_table.columns.tolist())

            # Extract the query tuple (example: 2nd row)
            query_tuple = input_table[query_cols].iloc[query_index, :].values
            candidate_col_indices = []
            for col_ in candidate_cols:
                try:
                    candidate_col_indices.append(datalake_tabs_header[candidate_name].index(col_))
                except:
                    close_matches = difflib.get_close_matches(col_, datalake_tabs_header[candidate_name], n=1)
                    if len(close_matches) > 0:
                        candidate_col_indices.append(datalake_tabs_header[candidate_name].index(close_matches[0]))
                    else:
                        continue

            if len(candidate_col_indices) == 0:
                print(f"missing {candidate_cols}")
                continue

            query_set_vals = [val_ for attr_, val_ in input_table[query_cols].iloc[query_index, :].items() if attr_!=missing_column]
            
            start_time = time.time()

            # Use the whole table as candidate tuples (excluding the query tuple)
            candidate_df = datalake_tabs[candidate_name]
            raw_candidate_df = candidate_df.copy()
            if len(candidate_df)>10000:
                candidate_df = find_rows_with_most_overlap_dask(candidate_df, query_set_vals)

            candidate_tuples = candidate_df.iloc[:,candidate_col_indices]
            # Find the top-5 most similar tuples to the query_tuple
            topk = mr.find_topk_similar_tuples(query_tuple, candidate_tuples, k=3)
            original_indices_selected = [(idx, score) for score, tup, idx in topk]

            cur_search_res = [(raw_candidate_df.iloc[orig_idx, map_id], candidate_name, orig_score) for orig_idx, orig_score in original_indices_selected]

            if query_id not in output_res:
                output_res[query_id] = cur_search_res
            else:
                output_res[query_id].extend(cur_search_res)

            # print('111', cur_search_res)
        # break

        end_time = time.time()
        all_search_time += end_time - start_time

        if missing_type=='cate':
            categorical_search_time += end_time-start_time 
        elif missing_type=='num':
            numerical_search_time += end_time-start_time

        print(f"Processing {all_cnt + 1} of {len(query_table_col_df)} done..." )
        all_cnt += 1

    print(f"Total search time is {all_search_time}")

    trans_output_res = aggregate_scores(output_res)
    sorted_output_res = sort_aggregated_scores(trans_output_res)

    return sorted_output_res, trans_output_res, all_search_time, categorical_search_time, numerical_search_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process table data.")
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--raw_benchmark', type=str, required=True, 
                        help="The benchmark that is created to build the index and so on. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    args = parser.parse_args()
    current_benchmark = args.benchmark
    raw_benchmark = args.raw_benchmark


    benchmark_path = os.path.abspath(f"../../{current_benchmark}_benchmark/")

    query_table_col_df = pd.read_csv(f'{benchmark_path}/missing_query/missing_tab_row_col.csv')
    impute_path = os.path.join(benchmark_path, 'impute_res')
    os.makedirs(impute_path, exist_ok=True)

    file_path = os.path.abspath(f"../test_stats/{raw_benchmark}_benchmark_result_by_santos_full.pkl")
    with bz2.BZ2File(file_path, "rb") as file:
        candidate_map_res = pickle.load(file)

    output_res, output_res_, out_time,  categorical_search_time, numerical_search_time \
         = process_query_table(query_table_col_df, benchmark_path, candidate_map_res)

    writer_path = os.path.join(impute_path, f'cesid_search.pkl')
    # Save the results
    with open(writer_path, 'wb') as f:     
        pickle.dump([output_res, output_res_], f)


    print(f'end processing and search for benchmark {current_benchmark} time {out_time} for {len(query_table_col_df)} missing queries...')