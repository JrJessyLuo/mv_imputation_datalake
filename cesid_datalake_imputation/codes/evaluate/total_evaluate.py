import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from math import sqrt
import re
import argparse
import os
import glob
import pickle

def clean_numeric_string(value):
    # Remove commas and any non-numeric characters (except for the decimal point)
    try:
        return float(re.sub(r'[^\d.]', '', str(value)))
    except:
        return np.nan


def evaluate_impute_res(ground_truth, method_res, missing_query_type_dict, query_tab_dict, scenario_keys, dtype='num'):
    results = []
    percentile_ranges = ['all', "s1", "s2", "s3"]
    common_key_dict = {'all': ground_truth.keys(), "s1":scenario_keys[0],\
        "s2":scenario_keys[1], "s3":scenario_keys[-1]}
    # Iterate over percentile ranges
    for percentile_range in percentile_ranges:
        
        current_common_keys = common_key_dict[percentile_range]
        print(f'Processing range: {percentile_range}, number of values: {len(current_common_keys)}, data type: {dtype}')

        for method_name, predictions in method_res.items():

            accuracy_list = []
            num_val_list_gt, num_val_list_pred = [], []
            wrong_cnt = 0

            for query, gt_val in tqdm(ground_truth.items(), total=len(ground_truth)):
                # query_tab = query.split(' || ')[0]
                # if query_tab not in current_common_keys:
                #     continue
                if query not in current_common_keys:continue
                

                # Handle categorical columns
                if dtype == 'cate' and missing_query_type_dict[query] == 'num':
                    continue

                # Handle numerical columns
                if dtype == 'num':
                    if missing_query_type_dict[query] in ['cate','text']:
                        continue
                    try:
                        gt_val = round(float(gt_val), 2)
                    except:
                        continue

                if query in predictions:
                    if dtype == 'num':
                        preds = [round(float(_), 2) if isinstance(_, (int, float, np.integer, np.floating)) or (isinstance(_, str) and _.replace('.', '', 1).isdigit()) else np.nan for _ in [predictions[query]]]
                    else:
                        preds = [_ for _ in [predictions[query]]]


                    if not preds:
                        preds = [np.nan if dtype == 'num' else '']
                        wrong_cnt += 1

                    if dtype == 'num':
                        tab_name, col_id = query.split(' || ')[0], int(query.split(' || ')[-2])
                        current_col = query_tab_dict[tab_name].iloc[:, col_id]
                        max_, min_, mean_ = current_col.apply(clean_numeric_string).max(), current_col.apply(clean_numeric_string).min(), current_col.apply(clean_numeric_string).mean()

                        if np.isnan(preds[0]):
                            if method_name not in predict_method_names:continue
                            elif not np.isnan(gt_val):preds=[max_]

                        cleaned_pred = clean_numeric_string(preds[0])
                        cleaned_gt_val = clean_numeric_string(gt_val)

                        if max_ == min_:
                            normalized_pred = 0
                            normalized_gt_val = 0
                        else:
                            normalized_pred = (cleaned_pred - min_) / (max_ - min_)
                            normalized_gt_val = (cleaned_gt_val - min_) / (max_ - min_)

                        if not np.isnan(float(gt_val)):
                            accuracy_list.append(normalized_pred == normalized_gt_val)
                            num_val_list_gt.append(normalized_gt_val)
                            num_val_list_pred.append(normalized_pred)


                    else:
                        accuracy_list.append(preds[0] == gt_val)
                else:
                    accuracy_list.append(0)

            # Aggregate results for the current method and percentile range
            accuracy = np.mean(accuracy_list)

            if len(accuracy_list)==0:
                continue
            if dtype == 'num':      
                mae = mean_absolute_error(num_val_list_gt, num_val_list_pred)
                rmse = sqrt(mean_squared_error(num_val_list_gt, num_val_list_pred))
                result = {
                    'method_name': method_name,
                    'percentile_range': percentile_range,
                    'accuracy': accuracy,
                    'mae': mae,
                    'rmse': rmse,
                    'wrong_count': wrong_cnt,
                    'total': len(accuracy_list)
                }
            else:
                result = {
                    'method_name': method_name,
                    'percentile_range': percentile_range,
                    'accuracy': accuracy,
                    'mae': -1,
                    'rmse': -1,
                    'wrong_count': wrong_cnt,
                    'total': len(accuracy_list)
                }

            # Append the result to the list
            results.append(result)

    # Convert the results into a pandas DataFrame for better visualization
    results_df = pd.DataFrame(results)

    return results_df

def obtain_gt_from_dataframe(input_dataframe):
    ground_truth, missing_query_type_dict = {}, {}
    for idx, row in input_dataframe.iterrows():
        query = f"{row['table_path']} || {row['column_name']} || {row['column_id']} || {row['row_id']}"
        ground_truth[query] = row['gt_val']
        missing_query_type_dict[query] = row['column_type']
    return ground_truth, missing_query_type_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process table data.")
    # Add arguments for benchmark and datatype
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    args = parser.parse_args()

    benchmark = args.benchmark

    benchmark_dir = os.path.abspath(f"../../{benchmark}_benchmark/")
    query_tab_dict = {}
    for f in glob.glob(os.path.join(benchmark_dir,'injected_missing_query','*.csv')):
        tab_name = f.split('/')[-1]
        query_tab_dict[tab_name] = pd.read_csv(f)
    missing_query_col_df = pd.read_csv(os.path.join(benchmark_dir, 'missing_query', 'missing_tab_row_col.csv'))
    ground_truth, missing_query_type_dict = obtain_gt_from_dataframe(missing_query_col_df)

    search_res_path = os.path.join(benchmark_dir, 'impute_res', f"cesid_search.pkl")
    search_keys = list(pickle.load(open(search_res_path,'rb'))[0].keys())
    est_enhance_path = os.path.abspath(f'../estimation/tmp_files/{benchmark}_enhance.pkl')
    est_enhance_cols = list(pickle.load(open(est_enhance_path,'rb')).keys())
    estimate_keys = [_ for _ in ground_truth.keys() if ' || '.join(_.split(" || ")[:-1]) in est_enhance_cols]
    other_keys = [_ for _ in ground_truth.keys() if _ not in search_keys and _ not in estimate_keys]
    print(len(search_keys), len(estimate_keys), len(other_keys))


    total_res_path = os.path.join(benchmark_dir, 'impute_res', f"cesid.pkl")
    method_res = {'cesid':pickle.load(open(total_res_path,'rb'))}

    num_res_df = evaluate_impute_res(ground_truth, method_res, missing_query_type_dict, query_tab_dict,\
         [search_keys, estimate_keys, other_keys], dtype='num')
    cate_res_df = evaluate_impute_res(ground_truth, method_res, missing_query_type_dict, query_tab_dict, \
        [search_keys, estimate_keys, other_keys], dtype='cate')

    print(other_keys)

    print(num_res_df)
    print(cate_res_df)

