import random
import warnings
import pandas as pd
import numpy as np
from scipy.stats import mode
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import collections
from collections import Counter
import argparse
import os
import pandas as pd
import bz2
import pickle
import tqdm
from tqdm import tqdm
import time
import fill_missing_values
from fill_missing_values import baseline_inpute
import difflib

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

def change_data_format(input_test_df, bound_, missing_col_idx, revert=False):
    test_df = input_test_df.copy()
    current_string_cols = [i for i in range(test_df.shape[1]) if test_df.iloc[:, i].dtype == object]
   
    encode_dict = {}
    dim_valid_sample_dict = {}

    for col in current_string_cols:
        # Get the unique values, excluding NaN
        unique_vals = test_df.iloc[:, col].dropna().unique()
        
        # Create a dictionary to map each unique value to an integer index
        unique_val_dict = {unique_vals[i]: i for i in range(len(unique_vals))}
        
        # Apply the dictionary to encode the column, handling NaN values separately
        test_df.iloc[:, col] = test_df.iloc[:, col].apply(lambda x: unique_val_dict.get(x, np.nan))
        
        # Store the encoding dictionary for future reference
        encode_dict[col] = unique_val_dict
  
    if not revert:
        dim_valid_sample_dict[missing_col_idx] = [test_df, bound_, encode_dict]
    else:
        dim_valid_sample_dict[missing_col_idx] = [input_test_df, bound_, encode_dict]
    return dim_valid_sample_dict


def process_query_table(query_table_col_df, benchmark_path):
    table_column_row_idx_dict, merged_table_dict = {}, {}
    _, query_tab_cache, query_tab_gt_val_dict = process_query_table_error(query_table_col_df, benchmark_path)
    print(f'{len(query_tab_cache)} query tables loading done...')

    for idx, row in tqdm(query_table_col_df.iterrows(), total=len(query_table_col_df)):
        table_name = row['table_path']
        input_table = query_tab_cache[table_name].copy()
        bound_ = len(input_table)

        query_long_index = f"{table_name} || {row['column_name']} || {row['column_id']}"
        if query_long_index not in table_column_row_idx_dict:
            table_column_row_idx_dict[query_long_index] = [row['row_id']]
        else:
            table_column_row_idx_dict[query_long_index].append(row['row_id'])
            continue

        input_table = input_table.fillna(np.nan)

        merged_table_dict[query_long_index] = [input_table, bound_]

    return merged_table_dict, table_column_row_idx_dict, query_tab_gt_val_dict

def process_query_table_error(query_table_col_df, benchmark_path):
    query_missing_dim = {}
    query_tab_cache = {}

    for query_tab in query_table_col_df['table_path'].unique().tolist():
        table_path = os.path.join(benchmark_path, "injected_missing_query", query_tab)
        if not os.path.exists(table_path):
            continue
        input_table = pd.read_csv(table_path, encoding='latin1')
        query_tab_cache[query_tab] = input_table
    print(f'{len(query_tab_cache)} query tables loading done...')

    gt_val_dict = {}

    for idx, row in tqdm(query_table_col_df.iterrows(), total=len(query_table_col_df)):
        table_name = row['table_path']
        input_table = query_tab_cache[table_name]

        row_idx, col_idx = int(row['row_id']), int(row['column_id'])

        query_long_index = f"{table_name} || {row['column_name']} || {row['column_id']}"
        if query_long_index not in gt_val_dict:
            gt_val_dict[query_long_index] = {}
        gt_val_dict[query_long_index][row_idx] = row['gt_val']


        if table_name not in query_missing_dim:
            query_missing_dim[table_name] = []
        if col_idx not in query_missing_dim[table_name]:
            query_missing_dim[table_name].append(col_idx)

    return query_missing_dim, query_tab_cache, gt_val_dict

def filter_valid_key(row, saved_dict):
    key_ = f"{row['table_path']} || {row['column_name']} || {row['column_id']} || {row['row_id']}"
    if key_ in saved_dict.keys():
        val_ = saved_dict[key_]
        if len(val_)==0:
            return 0
        else:
            return 1
    return 0



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Process table data.")
    # Add arguments for benchmark and datatype
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--method', type=str, default='mice/missforest', 
                        help="The estimation based method to use 'numerical_method,categorical_method'. \
                            The recommendations would be mice for numerical values, missforest for categorical values, the same as the default value.")
    args = parser.parse_args()
    current_benchmark = args.benchmark
    current_method_lst = args.method.split('/')
    num_method, cate_method = current_method_lst[0], current_method_lst[-1]

    benchmark_path = os.path.abspath(f'../../{current_benchmark}_benchmark/')
    query_table_col_df = pd.read_csv(f'{benchmark_path}/missing_query/missing_tab_row_col.csv')

    tmp_path = os.path.abspath("./tmp_files")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path, exist_ok=True)
    extended_output_row_dict = pickle.load(open(os.path.join(tmp_path, f'{current_benchmark}_enhance.pkl'),'rb'))
    saved_fpath = f"{benchmark_path}/impute_res/{current_benchmark}_{num_method}_{cate_method}_est.pkl"


    # query_table_col_df['is_exists'] = query_table_col_df.apply(lambda row: filter_valid_key(row, extended_output_row_dict), axis=1)
    # query_table_col_df = query_table_col_df[query_table_col_df['is_exists']==0].drop(columns=['is_exists'])
    print(f'we need to impute total {len(query_table_col_df)} rows')

    merged_table_dict, table_column_row_idx_dict, query_tab_gt_val_dict =  process_query_table(query_table_col_df, benchmark_path)

    output_raw_dict=  {}
    error_impute = 0
    total_time_cost1 = 0

    sampled_items = merged_table_dict.items()
    # Iterate over the sampled items
    for query_long_index, merged_candidate_infos in tqdm(sampled_items, total=len(sampled_items)):
        merged_data, raw_dataset_size,  = merged_candidate_infos
        missing_col_name = query_long_index.split(' || ')[-2]
        
        try:
            missing_col_idx = merged_data.columns.get_loc(missing_col_name)
        except:
            missing_col_name = get_single_column(missing_col_name, merged_data.columns)
            missing_col_idx = merged_data.columns.tolist().index(missing_col_name)

        if ['datawig'] in current_method_lst:
            filter_row_feat_dict = change_data_format(merged_data, raw_dataset_size, missing_col_idx, True)
        else:
            filter_row_feat_dict = change_data_format(merged_data, raw_dataset_size, missing_col_idx)

        # try:
        for col_, unioned_infos in filter_row_feat_dict.items():
            big_table, bound, cate_encode_dict = unioned_infos
            if query_long_index in extended_output_row_dict:
                big_table = extended_output_row_dict[query_long_index]
            big_table_complete = big_table.copy()

            try:
                big_table.iloc[:bound, missing_col_idx] = big_table.iloc[:bound, missing_col_idx].astype(float)
                col_dtype = 'num'
                min_val, max_val = big_table.iloc[:bound, missing_col_idx].min(), big_table.iloc[:bound, missing_col_idx].max()
                current_method = num_method
            except:
                common_vals = big_table.iloc[:bound, missing_col_idx].unique().tolist()
                col_dtype = 'cate'
                current_method = cate_method

            for cur_row_idx in table_column_row_idx_dict[query_long_index]:
                # print(query_long_index, cur_row_idx, query_tab_gt_val_dict[query_long_index])
                big_table_complete.iloc[cur_row_idx, col_] = query_tab_gt_val_dict[query_long_index][cur_row_idx]


            if current_method not in ['datawig']:
                basic_columns = big_table.columns
                impute_time_cost1, impute_res1 = baseline_inpute(big_table.iloc[:bound], col_, method=current_method)
                impute_res1 = pd.DataFrame(impute_res1, columns=basic_columns)
            elif current_method in ['datawig']:
                basic_columns = big_table.columns
                method_related_missing_col = [missing_col_name]
                impute_time_cost1, impute_res1 = baseline_inpute(big_table.iloc[:bound], method_related_missing_col, method=current_method)
                impute_res1 = pd.DataFrame(impute_res1, columns=basic_columns)

            if col_ in cate_encode_dict:
                raw_key_val_dict = {val:key for key, val in cate_encode_dict[col_].items()}
            else:
                raw_key_val_dict = None

            total_time_cost1 += impute_time_cost1

            for row_idx in table_column_row_idx_dict[query_long_index]:
                impute_key = f"{query_long_index} || {row_idx}"
                try:
                    output_raw_dict[impute_key]  = raw_key_val_dict[int(impute_res1.iloc[row_idx, col_])] if raw_key_val_dict is not None else impute_res1.iloc[row_idx, col_]
                except:           
                    print('I randomly select one of the value using 0...')
                    output_raw_dict[impute_key]  = raw_key_val_dict[0] if raw_key_val_dict is not None else impute_res1.iloc[row_idx, col_]



    print(f'before total saved count {len(output_raw_dict)}')
    pickle.dump(output_raw_dict, open(saved_fpath,'wb') )
