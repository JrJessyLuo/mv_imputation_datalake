import os
import glob
import pandas as pd
import argparse
import random
import numpy as np

def is_numerical_categorical_col(input_col):
    try:
        input_col = input_col.astype(float)
        return 'num'
    except:
        return 'cate'


def convert_dataframe_aslist(df, tab_name):
    all_lst = []
    columns_dict = {col: [idx,is_numerical_categorical_col(df[col])] for idx, col in enumerate(df.columns)}

    for idx, row in df.iterrows():
        for key_, val_ in row.to_dict().items():
            if pd.isna(val_):continue
            cur_dict = {'table_path': tab_name, "column_name": key_, "column_id": columns_dict[key_][0], "column_type": columns_dict[key_][-1], "row_id":idx, "gt_val": val_}
            all_lst.append(cur_dict)
    return all_lst

def convert_lst_dataframe(lst, df, tab_name):
    for _ in lst:
        if _['table_path']!=tab_name:continue
        row_id = _['row_id']
        column_id = _['column_id']
        df.iloc[row_id, column_id] = np.nan

    return df


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process table data.")
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--missing_cnt', type=int, default=100, 
                        help="You can set the amount of missing values")
    args = parser.parse_args()

    cur_benchmark = args.benchmark
    missing_cnt = args.missing_cnt

    query_tab_path = os.path.abspath(f"../../{cur_benchmark}_benchmark/query/")
    missing_value_records = os.path.abspath(f"../../{cur_benchmark}_benchmark/missing_query/")
    if not os.path.exists(missing_value_records):
        os.makedirs(missing_value_records, exist_ok=True)

    missing_value_tmp = os.path.abspath(f"../../{cur_benchmark}_benchmark/injected_missing_query/")
    if not os.path.exists(missing_value_tmp):
        os.makedirs(missing_value_tmp, exist_ok=True)

    all_vals, all_raw_dict = [], {}
    for f in glob.glob(os.path.join(query_tab_path,'*.csv')):
        df = pd.read_csv(f)
        tab_name = f.split('/')[-1]
        cur_vals = convert_dataframe_aslist(df, tab_name)
        all_vals.extend(cur_vals)
        all_raw_dict[tab_name] = df

    if not os.path.exists(os.path.join(missing_value_records, 'missing_tab_row_col.csv')):
        sampled_missing_vals =  random.sample(all_vals, missing_cnt)
        # save the ground truth for each missing value
        missing_query_df = pd.DataFrame.from_records(sampled_missing_vals)
        missing_query_df.to_csv(os.path.join(missing_value_records, 'missing_tab_row_col.csv'), index=False)
    else:
        sampled_missing_vals = pd.read_csv(os.path.join(missing_value_records, 'missing_tab_row_col.csv')).to_records(index=False)

    # inject the missing values to the raw dataframe
    injected_missing_dict = {}
    for tab_name, df in all_raw_dict.items():
        injected_missing_dict[tab_name] = convert_lst_dataframe(sampled_missing_vals, df, tab_name)
        injected_missing_dict[tab_name].to_csv(os.path.join(missing_value_tmp, f"{tab_name}"), index=False)