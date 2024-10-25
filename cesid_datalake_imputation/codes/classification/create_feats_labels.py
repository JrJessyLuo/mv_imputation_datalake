import pandas as pd
import glob
import os
import tqdm
from tqdm import tqdm
import pickle
import argparse
from multiprocessing import Pool
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import collections
from collections import Counter

# Determine if the input table consists entirely of numerical attributes
def is_numerical_table(input_table, threshold=1.0):
    numerical_cols = input_table.select_dtypes(include=['number']).columns
    return len(numerical_cols) / len(input_table.columns) >= threshold \
    , len(numerical_cols)/len(input_table.columns)

def get_csv_header(file_path):
    with open(file_path, 'r') as file:
        header = next(file).strip().split(',')
    return header

def presave_datalake_header(datalake_header_path, datalake_fls):
    if not os.path.exists(datalake_header_path):
        datalake_headers = {}
        for f in tqdm(datalake_fls, total = len(datalake_fls)):
            datalake_tab_header = get_csv_header(f)
            tab_name = f.split('/')[-1]
            datalake_headers[tab_name] = datalake_tab_header
        pickle.dump(datalake_headers, open(datalake_header_path, 'wb'))
    else:
        datalake_headers = pickle.load(open(datalake_header_path,'rb'))

    return datalake_headers

def preload_datalake_content(datalake_fls):
    datalake_df_content = {}
    for f in tqdm(datalake_fls, total = len(datalake_fls)):
        try:
            datalake_cur_df = pd.read_csv(f, encoding='latin1')
            tab_name = f.split('/')[-1]
            datalake_df_content[tab_name] = datalake_cur_df
        except:
            print('error')
            pass
    return datalake_df_content

def value_overlap(query_values, df_values):
    overlap = query_values.intersection(df_values)
    return len(overlap)/len(query_values)

def flatten_dataframe_values(df):
    return set(df.values.flatten())

def find_largest_overlap_parallel(query_df, dataframes, num_workers=4):
    query_values = flatten_dataframe_values(query_df)

    with Pool(num_workers) as pool:
        # Pre-flatten all dataframes and compute overlaps in parallel
        flattened_dfs = pool.map(flatten_dataframe_values, dataframes.values())
        overlap_counts = pool.starmap(value_overlap, [(query_values, df) for df in flattened_dfs])

    # Find the dataframe with the maximum overlap
    max_overlap_idx = overlap_counts.index(max(overlap_counts))
    best_df_name = list(dataframes.keys())[max_overlap_idx]
    return best_df_name, max(overlap_counts), min(overlap_counts), np.mean(overlap_counts)

def count_matching_lists(target_list, collection_of_lists, match_type='exact'):
    match_count = 0
    matching_lists = []

    for key, lst in collection_of_lists.items():
        if match_type == 'exact':
            if lst == target_list:
                match_count += 1
                matching_lists.append(key)
        elif match_type == 'unordered':
            if sorted(lst) == sorted(target_list):
                match_count += 1
                matching_lists.append(key)
        elif match_type == 'subset':
            if set(target_list).issubset(set(lst)):
                match_count += 1
                matching_lists.append(key)
    
    return match_count, matching_lists

def construct_tab_lvl_feats(query_tab_path, saved_fpath, datalake_headers, datalake_df_content, table_schema_matches_path):
    total_datalake_fls = glob.glob(os.path.join(query_tab_path, '*.csv'))

    tab_feature_list = []
    tab_matched_tabs = {}



    for f in tqdm(total_datalake_fls, total=len(total_datalake_fls)):
        tab_name = f.split('/')[-1]

        input_tab = pd.read_csv(f,encoding='latin1')
        row_cnt, col_cnt = len(input_tab), len(input_tab.columns)
        query_tab_header = get_csv_header(f)
        numerical_flag, numerical_ratio = is_numerical_table(input_tab)
        matched_cnt, matched_list = count_matching_lists(query_tab_header, datalake_headers, match_type='unordered')
        tab_matched_tabs[tab_name] = matched_list

        if len(matched_list)>0:
            tmp_datalake_df_content = {key:val for key,val in datalake_df_content.items() if key in matched_list}
        else:
            tmp_datalake_df_content = datalake_df_content

        max_tab_name, max_overlap_ratio, min_overlap_ratio, mean_overlap_ratio = find_largest_overlap_parallel(input_tab, tmp_datalake_df_content)

        tab_feature_list.append([tab_name, row_cnt, col_cnt, numerical_ratio, matched_cnt, max_overlap_ratio, min_overlap_ratio, mean_overlap_ratio, max_tab_name])


    tab_feature_df = pd.DataFrame(tab_feature_list, columns=['Tab Name', 'Row Cnt', 'Column Cnt', 'Numerical Ratio', 'Matched Count', 'Max Overlap Ratio',\
        'Min Overlap Ratio', 'Mean Overlap Ratio', 'Max Overlap Table Name'])
    tab_feature_df.to_csv(saved_fpath, index=False)
    pickle.dump(tab_matched_tabs, open(table_schema_matches_path, 'wb'))
    return tab_feature_df, tab_matched_tabs

def count_column_names(col_name, datalake_headers):
    return len([_ for _ in datalake_headers.values() if col_name in _])

def calculate_overlap_ratio(value_set, col_values_set):
    # Perform the set intersection and calculate overlap ratio
    intersection = value_set & col_values_set
    return len(intersection) / len(value_set)

def process_dataframe_columns(df, value_set):
    # Precompute unique sets for all columns
    unique_column_values = {col: set(df[col].dropna().unique()) for col in df.columns}

    # Store column names and their overlap ratios
    column_overlap = []
    
    for col, col_values_set in unique_column_values.items():
        if col_values_set:  # Ensure the column has values
            overlap_ratio = calculate_overlap_ratio(value_set, col_values_set)
            column_overlap.append((col, overlap_ratio))

    return column_overlap

# construct the row overlap ratio
def max_containment_ratio(target_set, list_of_sets):
    max_ratio = 0
    best_set = None

    for value_set in list_of_sets:
        intersection = target_set.intersection(value_set)
        ratio = len(intersection) / len(target_set)  # Containment ratio

        if ratio > max_ratio:
            max_ratio = ratio
            best_set = value_set

    return best_set, max_ratio


def find_best_overlap_column_parallel(dataframes, value_set, col_name):
    value_set = set(value_set)  # Ensure value_set is a set

    best_column = None
    best_overlap_ratio = 0

    # List to store all overlap ratios for statistical calculations
    overlap_ratios = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_df = {executor.submit(process_dataframe_columns, df, value_set): df for df in dataframes}
        
        for future in as_completed(future_to_df):
            df = future_to_df[future]
            try:
                column_overlap = future.result()
                # Store all overlap ratios
                overlap_ratios.extend([overlap_ratio for _, overlap_ratio in column_overlap if overlap_ratio>0])

                # Check for the best column across all results
                for col, overlap_ratio in column_overlap:
                    if overlap_ratio > best_overlap_ratio:
                        best_column = col

            except Exception as e:
                print(f"Error processing dataframe: {e}")

    # Calculate max, min, and mean overlap ratios
    if overlap_ratios:
        max_overlap = max(overlap_ratios)
        min_overlap = min(overlap_ratios)
        mean_overlap = np.mean(overlap_ratios)
        overlap_cnt = len(overlap_ratios)
    else:
        max_overlap = min_overlap = mean_overlap = overlap_cnt = 0
    if best_column == col_name:flag = 1
    else:flag = 0

    return max_overlap, min_overlap, mean_overlap, overlap_cnt, flag

def construct_column_lvl_feats(col_level_df, query_tab_path, column_feats_saved_path, table_matches):
    col_feature_dict = {}
    for idx, row in tqdm(col_level_df.iterrows(), total = len(col_level_df)):
        tab_name = row['table_path']
        f = os.path.join(query_tab_path, row['table_path'])
        input_table = pd.read_csv(f,encoding='latin1')
        query_tab_header = get_csv_header(f)
        col_id = int(row['column_id'])
        col_name = row['column_name']

        identifier = f"{row['table_path']} || {row['column_name']} || {row['column_id']}"

        col_val_set = input_table.iloc[:, col_id].unique()
        col_name_hit_cnt = count_column_names(col_name, datalake_headers)

        missing_cnt = len(input_table.iloc[:, col_id].isna())
        missing_ratio = 1-input_table.iloc[:, col_id].count()/len(input_table)

        column_type = 1 if row['column_type']=='num' else 0
        matched_list = table_matches[tab_name]

        if len(matched_list)>0:
            tmp_datalake_df_content = {key:val for key,val in datalake_df_content.items() if key in matched_list}
        else:
            tmp_datalake_df_content = datalake_df_content

        max_overlap, min_overlap, mean_overlap, overlap_col_cnt, flag = find_best_overlap_column_parallel(tmp_datalake_df_content.values(), col_val_set, col_name)

        col_feature_dict[identifier] = [len(col_val_set), col_name_hit_cnt, max_overlap, min_overlap, mean_overlap, missing_cnt, missing_ratio, column_type]
        
        # _, max_overlap_ratio = find_largest_overlap_parallel(input_tab, tmp_datalake_df_content)
    pickle.dump(col_feature_dict, open(column_feats_saved_path,'wb'))

    return col_feature_dict

"""
    row level features by iterating each row to calculate the max overlap ratio
"""
def construct_row_level_feats(row_level_df, table_feats, query_tab_path, datalake_tab_path, saved_fpath):
    max_table_dict = {}
    for idx, row in table_feats.iterrows():
        query_tab_name = row['Tab Name']
        candidate_tab_name = row['Max Overlap Table Name']
        max_table_dict[query_tab_name] = candidate_tab_name

    query_tab_df, datalake_df = {}, {}
    row_level_df['max_containment_row'] = -1
    for idx, row in row_level_df.iterrows():
        tab_name = row['table_path']
        if tab_name in datalake_df:
            input_tab_df = query_tab_df[tab_name]
        else:
            input_tab_df = pd.read_csv(os.path.join(query_tab_path, tab_name))
            query_tab_df[tab_name] = input_tab_df
        candidate_tab_name = max_table_dict[tab_name]
        if candidate_tab_name not in datalake_df:
            candidate_tab_df = pd.read_csv(os.path.join(datalake_tab_path, candidate_tab_name))
            datalake_df[candidate_tab_name] = candidate_tab_df
        else:
            candidate_tab_df = datalake_df[candidate_tab_name]
        input_tab_query_row = set(input_tab_df.iloc[int(row['row_id']),:].values)
        _, max_containment_val = max_containment_ratio(input_tab_query_row, [set(row.values) for _, row in candidate_tab_df.iterrows()])
        row_level_df.at[idx, 'max_containment_row'] = max_containment_val

    row_level_df.to_csv(saved_fpath, index=False)
    return row_level_df

def construct_train_labels(search_res, predict_res, ground_truth_df, train_label_path):
    impute_source_dict = {}
    search_hit_cnt = 0
    for idx, row in ground_truth_df.iterrows():
        query = f"{row['table_path']} || {row['column_name']} || {row['column_id']} || {row['row_id']}"
        dtype = row['column_type']
        gt_val = row['gt_val']
        impute_source_dict[query] = 'estimate'
        if query in search_res:
            if dtype == 'num':
                search_val, predict_val = search_res[query][0], predict_res[query]
                try:
                    if float(search_val)==float(gt_val):
                        impute_source_dict[query]= 'search'
                        search_hit_cnt += 1
                except:
                    print('search error!!!')
                    print(search_val)
                    
            elif dtype == 'cate':
                if search_res[query][0]==gt_val:
                    predict_val = search_res[query][0] 
                    impute_source_dict[query] = 'search'
                    search_hit_cnt += 1 
    pickle.dump(impute_source_dict, open(train_label_path, 'wb'))
    print(f"search with {search_hit_cnt} in {len(ground_truth_df)}")
    return impute_source_dict

             
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process table data.")
    # Add arguments for benchmark and datatype
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--train_benchmark', type=str, required=True, 
                        help="The benchmark used to create the training dataset. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    args = parser.parse_args()

    benchmark = args.benchmark
    train_benchmark = args.train_benchmark

    basic_dir = os.path.abspath(f"../../{benchmark}_benchmark/")

    query_tab_path = os.path.join(basic_dir, "injected_missing_query")
    datalake_tab_path = os.path.join(basic_dir, "datalake")
    datalake_fls = glob.glob(os.path.join(datalake_tab_path, '*.csv'))

    if not os.path.exists(os.path.abspath(f"./classifier_tmp")):
        os.mkdir(os.path.abspath(f"./classifier_tmp"))

    datalake_header_path = os.path.abspath(f"./classifier_tmp/{benchmark}_datalake_headers.pkl")
    table_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{benchmark}_tbl_summary.csv")
    table_schema_matches_path = os.path.abspath(f"./classifier_tmp/{benchmark}_tab_matches.pkl")
    column_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{benchmark}_col_summary.pkl")

    datalake_headers = presave_datalake_header(datalake_header_path, datalake_fls)
    datalake_df_content = preload_datalake_content(datalake_fls)

    if not os.path.exists(table_lvl_feats_path):
        table_feats, table_matches = construct_tab_lvl_feats(query_tab_path, table_lvl_feats_path, datalake_headers, datalake_df_content, table_schema_matches_path)
    else:
        table_feats = pd.read_csv(table_lvl_feats_path)
        table_matches = pickle.load(open(table_schema_matches_path, 'rb'))

    missing_query_col_df = pd.read_csv(os.path.join(basic_dir, 'missing_query', 'missing_tab_row_col.csv'))
    col_level_df = missing_query_col_df[['table_path', 'column_name', 'column_id', 'column_type']].drop_duplicates()

    if not os.path.exists(column_lvl_feats_path):
        column_feats = construct_column_lvl_feats(col_level_df, query_tab_path, column_lvl_feats_path, table_matches)

    
    row_level_df = missing_query_col_df[['table_path', 'column_name', 'column_id',  'row_id']].drop_duplicates()
    row_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{benchmark}_row_summary.csv")

    if not os.path.exists(row_lvl_feats_path):
        row_feats_df = construct_row_level_feats(row_level_df, table_feats, query_tab_path, datalake_tab_path, row_lvl_feats_path)

    train_dir = os.path.abspath(f"../../{train_benchmark}_benchmark/")
    missing_query_col_df = pd.read_csv(os.path.join(train_dir, 'missing_query', 'missing_tab_row_col.csv'))
    row_level_df = missing_query_col_df[['table_path',  'column_name', 'column_id',  'row_id']].drop_duplicates()
    row_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{train_benchmark}_row_summary.csv")
    train_label_path = os.path.abspath(f"./classifier_tmp/{train_benchmark}_train_labels.pkl")

    train_search_res_path = os.path.join(train_dir, 'impute_res', f"cesid_search.pkl")
    train_est_res_path = glob.glob(os.path.join(train_dir, 'impute_res', f"{train_benchmark}_*_est.pkl"))

    if os.path.exists(train_search_res_path):
        search_res = pickle.load(open(train_search_res_path, 'rb'))[0]
    else:
        print('lack of training data ....')
        assert True

    if len(train_est_res_path)==0:
        print('lack of training data ....')
        assert True
    else:
        predict_res = pickle.load(open(train_est_res_path[0], 'rb'))

    
    if not os.path.exists(row_lvl_feats_path):
        row_feats_df = construct_row_level_feats(row_level_df, table_feats, query_tab_path, datalake_tab_path, row_lvl_feats_path)

    if not os.path.exists(train_label_path):
        output_labels = construct_train_labels(search_res, predict_res, missing_query_col_df, train_label_path)

