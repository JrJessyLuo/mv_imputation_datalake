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
import utils
from utils import encoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def merge_multiple_feats(table_feats_df, column_feats, test_row_feats_df, labels=None):
    #     'Tab Name', 'Row Cnt', 'Column Cnt', 'Numerical Ratio', 'Matched Count',
    #    'Max Overlap Ratio', 'Min Overlap Ratio', 'Mean Overlap Ratio'

    # f"{table_name} || {col_name} || {col_id}":[len(col_val_set), col_name_hit_cnt, max_overlap, min_overlap, mean_overlap, missing_cnt, missing_ratio, column_type]
    # 'table_path', 'row_id', 'max_containment_row'
    test_row_feats_df = test_row_feats_df.rename(columns={'table_path': 'Tab Name', 'column_name': 'Col Name', 'column_id':'Col ID' })
    table_feats_df = test_row_feats_df.merge(table_feats_df, how='left', on=['Tab Name'])

    data_expanded = [
        key.split(" || ") + value  # Split key into three parts and append the value list
        for key, value in column_feats.items()
    ]

    columns = [
        "Tab Name", "Col Name", "Col ID",  
        'Column Value Unique Count',  
        'Overlap Column Count',             
        'Max Value Overlap Ratio',         
        'Min Value Overlap Ratio',             
        'Mean Value Overlap Ratio',              
        'Missing Count',               
        'Missing Ratio',
        'Column Type'                             
    ]
    column_lvl_df = pd.DataFrame(data_expanded, columns=columns)
    column_lvl_df = column_lvl_df.merge(table_feats_df, how='left', on=['Tab Name', 'Col Name' ])
    
    all_feats_df = column_lvl_df[['Column Value Unique Count',
       'Overlap Column Count', 'Max Value Overlap Ratio',
       'Min Value Overlap Ratio', 'Mean Value Overlap Ratio', 'Missing Count',
       'Missing Ratio', 'Row Cnt', 'Column Cnt', 'Numerical Ratio',
       'Matched Count', 'Max Overlap Ratio', 'Min Overlap Ratio',
       'Mean Overlap Ratio','max_containment_row', 'Column Type']]

    if labels:
        column_lvl_df['label'] = column_lvl_df.apply(lambda row: \
            1 if labels.get(f"{row['Tab Name']} || {row['Col Name']} || {row['Col ID_x']} || {row['row_id']}")=='search' else 0, axis=1)
        return all_feats_df, column_lvl_df['label']

    return all_feats_df, column_lvl_df[["Tab Name", "Col Name", "Col ID_x", "row_id"]]


def normalize_dataframe(train_feats_df, train_labels, test_feats_df):
    feat_encoder = encoder.Encoder('classification')
    mask = [len(train_feats_df.columns)-1]
    scaler = StandardScaler()


    feat_encoder.fit( train_feats_df, train_labels, mask)
    converted_train_feats = feat_encoder.transform_dataset(train_feats_df, list(range(len(train_feats_df.columns))))
    converted_test_feats = feat_encoder.transform_dataset(test_feats_df, list(range(len(test_feats_df.columns))))
    
    normalized_train_feats = scaler.fit_transform(converted_train_feats)
    normalized_test_feats = scaler.transform(converted_test_feats)
    return normalized_train_feats, train_labels, normalized_test_feats


def knn_classify(train_feats, train_labels, test_feats):
    train_feats, train_labels, test_feats = normalize_dataframe(train_feats, train_labels, test_feats)
    # Step 1: Split the training data to find the best 'k' using validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_feats, train_labels, test_size=0.2, random_state=42
    )

    # Step 2: Try different values of 'k' to find the best one
    best_k = None
    best_accuracy = 0
    for k in range(1, 10):  # Try k from 1 to 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_predictions = knn.predict(X_val)
        accuracy = accuracy_score(y_val, val_predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    # Step 3: Train the KNN classifier using the best 'k' on the full training data
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(train_feats, train_labels)

    # Step 4: Predict on the test features
    test_predictions = knn.predict(test_feats)

    return test_predictions

def combine_search_est_res(search_res, predict_res, test_identifiers, test_predicts):
    output_res_dict = {}
    for idx, row in test_identifiers.iterrows():
        query = f"{row['Tab Name']} || {row['Col Name']} || {row['Col ID_x']} || {row['row_id']}"
        label = test_predicts[idx]
        if label==1:
            # using search result
            output_res_dict[query] = search_res[query]
        else:
            # using estimate result
            output_res_dict[query] = predict_res[query]
    return output_res_dict


             
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

    if not os.path.exists(os.path.abspath(f"./classifier_tmp")):
        os.mkdir(os.path.abspath(f"./classifier_tmp"))

    table_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{benchmark}_tbl_summary.csv")
    column_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{benchmark}_col_summary.pkl")


    if not os.path.exists(table_lvl_feats_path):
        print('missing the table level feats')
        assert True
    else:
        table_feats_df = pd.read_csv(table_lvl_feats_path)
    #     'Tab Name', 'Row Cnt', 'Column Cnt', 'Numerical Ratio', 'Matched Count',
    #    'Max Overlap Ratio', 'Min Overlap Ratio', 'Mean Overlap Ratio'

    if not os.path.exists(column_lvl_feats_path):
        print('missing the column level feats')
        assert True
    else:
        column_feats = pickle.load(open(column_lvl_feats_path,'rb'))
        # f"{table_name} || {col_name} || {col_id}":[len(col_val_set), col_name_hit_cnt, max_overlap, min_overlap, mean_overlap, missing_cnt, missing_ratio, column_type]

    row_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{benchmark}_row_summary.csv")

    if not os.path.exists(row_lvl_feats_path):
        print('missing the row level feats')
        assert True
    else:
        test_row_feats_df = pd.read_csv(row_lvl_feats_path)
        # 'table_path', 'row_id', 'max_containment_row'

    test_all_feats, test_identifiers = merge_multiple_feats(table_feats_df, column_feats, test_row_feats_df)

    train_dir = os.path.abspath(f"../../{train_benchmark}_benchmark/")
    row_lvl_feats_path = os.path.abspath(f"./classifier_tmp/{train_benchmark}_row_summary.csv")
    train_label_path = os.path.abspath(f"./classifier_tmp/{train_benchmark}_train_labels.pkl")

    train_row_feats_df = pd.read_csv(row_lvl_feats_path)
    train_labels = pickle.load(open(train_label_path,'rb'))

    train_all_feats, train_labels = merge_multiple_feats(table_feats_df, column_feats, train_row_feats_df, train_labels)

    test_predicts  = knn_classify(train_all_feats, train_labels, test_all_feats)

    test_dir = os.path.abspath(f"../../{benchmark}_benchmark/")
    test_search_res_path = os.path.join(test_dir, 'impute_res', f"cesid_search.pkl")
    test_est_res_path = glob.glob(os.path.join(test_dir, 'impute_res', f"{benchmark}_*_est.pkl"))

    if os.path.exists(test_search_res_path):
        search_res = pickle.load(open(test_search_res_path, 'rb'))[0]
        search_res = {key:val[0] for key,val in search_res.items()}
    else:
        print(f'lack of training data .... in {test_search_res_path}')

    if len(test_est_res_path)==0:
        print(f'lack of training data .... in {test_est_res_path}')
    else:
        predict_res = pickle.load(open(test_est_res_path[0], 'rb'))

    test_output = combine_search_est_res(search_res, predict_res, test_identifiers, test_predicts)

    pickle.dump(test_output, open(os.path.join(test_dir, 'impute_res', f"cesid.pkl"),'wb'))
