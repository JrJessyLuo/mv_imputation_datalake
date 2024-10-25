import pandas as pd
import glob
import tqdm
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy, kurtosis, skew
import hnswlib
from sklearn.preprocessing import StandardScaler
from datasketch import MinHash, MinHashLSHEnsemble
import pickle
import os
import time
import utils
from utils import generalFunctions as genFunc
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    prepare_feature_extraction,
)
import os
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features
from collections import OrderedDict
initialise_pretrained_model(400)
initialise_nltk()
prepare_feature_extraction()
import re
from sentence_transformers import SentenceTransformer
import argparse

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # A multilingual model

def preprocess_column_name(column_name):
    # Split based on common delimiters
    tokens = re.split(r'[_\W]+', column_name)
    
    # Optionally: Remove or handle numeric tokens
    tokens = [token for token in tokens if not token.isdigit()]
    
    # Join tokens back into a string for embedding
    processed_name = ' '.join(tokens).strip()
    
    return processed_name

# Convert column names to embeddings using a transformer model
def convert_column_names_to_embeddings(column_names):
    

    # Preprocess each column name
    processed_column_names = [preprocess_column_name(name) for name in column_names]

    # Convert to embeddings
    embeddings = model.encode(processed_column_names)

    return embeddings

# Extract features from column values and combine with column name embeddings
def extract_features(col_values: list, column_name):
    features = OrderedDict()

    # start_time = time.time()
    # Extract features using existing methods
    # extract_word_embeddings_features(col_values, features)
    # cost_time = time.time() -start_time
    # print(f"{column_name} word embedding cost for {cost_time}")
    # start_time = time.time()
    infer_paragraph_embeddings_features(col_values, features, dim=400, reuse_model=True)
    # cost_time = time.time() -start_time
    # print(f"{column_name} paragraph embedding cost for {cost_time}")

    # start_time = time.time()
    # Convert the column name to an embedding
    col_name_feat_vec = convert_column_names_to_embeddings([column_name])
    # cost_time = time.time() -start_time
    # print(f"{column_name} column name embedding cost for {cost_time}")

    # Convert extracted features to a numpy array
    feature_vector = list(features.values())
    feature_vector = np.array(feature_vector)

    # Concatenate the column name embedding with the other features
    combined_feature_vector = np.concatenate((feature_vector.flatten(), col_name_feat_vec.flatten()))

    return combined_feature_vector


class ColumnLSHEnsembleIndex:
    def __init__(self, num_perm=128, threshold=0.05, num_part=32):
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_part = num_part
        self.lsh_ensemble = MinHashLSHEnsemble(threshold=self.threshold, num_perm=self.num_perm, num_part=self.num_part)
        self.index_data = []
        self.keys_set = set()  # To keep track of added keys

    def create_minhash(self, values):
        m = MinHash(num_perm=self.num_perm)
        for value in values:
            m.update(str(value).encode('utf-8'))
        return m
    
    def add_num_index_data(self, column_values, table_name, column_name, column_id):
        column_key = f"{table_name} || {column_name} || {column_id}"
        if column_key in self.keys_set:
            print(f"Duplicate key found and skipped: {column_key}")
            return  # Skip adding duplicate keys

        m = self.create_minhash(column_values)
        self.index_data.append((column_key, m, len(column_values)))
        self.keys_set.add(column_key)

    def add_str_index_data(self, col_values, table_name, column_name, column_id):
        column_key = f"{table_name} || {column_name} || {column_id}"
        if column_key in self.keys_set:
            print(f"Duplicate key found and skipped: {column_key}")
            return  # Skip adding duplicate keys

        m = self.create_minhash(col_values)
        self.index_data.append((column_key, m, len(col_values)))
        self.keys_set.add(column_key)

    def index(self):
        self.lsh_ensemble.index(self.index_data)
        print('lsh built done...')
        return self.lsh_ensemble


def query_jaccard(query_values, lsh_ensemble, num_perm=128 ):
    query_m = MinHash(num_perm=num_perm)

    for value in query_values:
        query_m.update(str(value).encode('utf-8'))

    if query_m is None:
        # MinHash for query column was not found, handle the error or rebuild it
        return []
    
    candidate_results = lsh_ensemble.query(query_m, len(query_m))

    return candidate_results

def compute_stats(values):
    _min = min(values)
    _max = max(values)
    _sum = sum(values)

    _mean = np.mean(values)

    x = values - _mean

    _variance = np.mean(x * x)

    if _variance == 0:
        _skew = 0
        _kurtosis = -3
    else:
        _skew = np.mean(x ** 3) / _variance ** 1.5
        _kurtosis = np.mean(x ** 4) / _variance ** 2 - 3

    return _mean, _variance, _skew, _kurtosis, _min, _max, _sum


def convert_numerical_feature(input_num_col):
    # Drop NaN values for certain calculations
    num_col = input_num_col.dropna()
    _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(num_col)

    # Features
    features = {}

    # 1. Number of values
    # features['num_values'] = len(num_col)

    # 2. Column entropy
    # features['column_entropy'] = entropy(num_col.value_counts(normalize=True), base=2)

    # 3. Fraction of values with unique content
    features['fraction_unique'] = num_col.nunique() / len(num_col)

    # 6. Mean and std. of the number of numerical characters in values
    num_char_lengths = num_col.astype(str).apply(len)
    features['mean_num_char'] = num_char_lengths.mean()
    features['std_num_char'] = num_char_lengths.std()

    features.update({
        '_mean': _mean,
        '_variance': _variance,
        '_kurtosis': _kurtosis,
        '_skew': _skew,
        '_min': _min,
        '_max': _max,
        '_sum': _sum
    })

    # Convert features dictionary to a numpy vector
    features_vector = np.array(list(features.values()))
    return features_vector

# judge whether is the numerical column
def is_numerical_col(df, col, digit_threshold=8):
    if df[col].count()/len(df[col])<0.1:
        return 0
    if pd.api.types.is_numeric_dtype(df[col]):
        # If the column is numerical, add it to valid_cols
        col_filtered = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if col_filtered.empty:
            return 0
        # If the column is numerical, add it to valid_cols
        max_digits = col_filtered.apply(lambda x: len(str(int(x))) if pd.notna(x) else 0).max()
        if max_digits > digit_threshold:
            return 0
        return 1
    else:
        try:
            df[col] = df[col].astype(float)
            return 1
        except:
            return 0
        
def clean_numerical_col(column):
    column = column.str.replace(',', '', regex=False)
    # Trim leading and trailing whitespaces
    column = column.str.strip()
    # Convert empty strings to NaN
    column = column.replace('', np.nan)
    
    # Try to convert to numeric, forcing errors to NaN
    column = pd.to_numeric(column, errors='coerce')
    return column

def get_topk_jaccard_cols(lsh_index, hnsw_index, scaler, meta_data, query_df, intent_col_id, intent_col_type, topk=50):
    if intent_col_type=='num':
        query_values = [str(round(float(val_),4)) for val_ in query_df.iloc[:, intent_col_id].dropna().values]
        query_feature_vec = convert_numerical_feature(clean_numerical_col(query_df.iloc[:, intent_col_id].astype(str)))
    elif intent_col_type=='cate':
        query_values = genFunc.preprocessListValues(query_df.iloc[:, intent_col_id])
        query_feature_vec = extract_features(list(set(query_values)), query_df.columns[intent_col_id])

    if not check_for_invalid_values(query_feature_vec):
        query_feature_vec = handle_invalid_values(query_feature_vec)
        print('nan vectors!!')

    jaccard_query_result = [r for r in query_jaccard(list(set(query_values)), lsh_index)]

    return jaccard_query_result

def get_topk_hnsw_cols_byindices(hnsw_index, specific_indices, scaler, meta_data, query_df, intent_col_id, intent_col_type, topk=50):
    subset_feature_vectors = hnsw_index.get_items(specific_indices)

    if intent_col_type=='num':
        query_values = [str(round(float(val_),4)) for val_ in query_df.iloc[:, intent_col_id].dropna().values]
        query_feature_vec = convert_numerical_feature(clean_numerical_col(query_df.iloc[:, intent_col_id].astype(str)))
    elif intent_col_type=='cate':
        query_values = genFunc.preprocessListValues(query_df.iloc[:, intent_col_id])
        query_feature_vec = extract_features(list(set(query_values)), query_df.columns[intent_col_id])

    if not check_for_invalid_values(query_feature_vec):
        query_feature_vec = handle_invalid_values(query_feature_vec)
        print('nan vectors!!')

    if scaler is not None:
        query_feature_vec = scaler.transform([query_feature_vec])
    else:
        query_feature_vec = [query_feature_vec]

    dim = subset_feature_vectors.shape[1]  # Get the dimension of the feature vectors
    temp_hnsw_index = hnswlib.Index(space='l2', dim=dim)  # Create a temporary HNSW index
    temp_hnsw_index.init_index(max_elements=len(specific_indices), ef_construction=200, M=16)

    temp_hnsw_index.add_items(subset_feature_vectors)
    temp_hnsw_index.set_ef(topk)
    
    labels, distances = temp_hnsw_index.knn_query(query_feature_vec, k=topk)

    topk_res = []
    for i in range(len(query_feature_vec)):
        lbl_, dis_ = labels[i], distances[i]
        for j in range(len(lbl_)):
            original_idx = specific_indices[lbl_[j]]  # Map back to the original index
            topk_res.append(meta_data[original_idx])

    return topk_res

def get_topk_jaccard_hnsw_cols(lsh_index, hnsw_index, scaler, meta_data, query_df, intent_col_id, intent_col_type, topk=50):
    if intent_col_type=='num':
        query_values = [str(round(float(val_),4)) for val_ in query_df.iloc[:, intent_col_id].dropna().values]
        query_feature_vec = convert_numerical_feature(clean_numerical_col(query_df.iloc[:, intent_col_id].astype(str)))
    elif intent_col_type=='cate':
        query_values = genFunc.preprocessListValues(query_df.iloc[:, intent_col_id])
        query_feature_vec = extract_features(list(set(query_values)), query_df.columns[intent_col_id])

    if not check_for_invalid_values(query_feature_vec):
        query_feature_vec = handle_invalid_values(query_feature_vec)
        print('nan vectors!!')

    jaccard_query_result = [r for r in query_jaccard(list(set(query_values)), lsh_index)]
    # print('-----',jaccard_query_result, list(set(query_values)),)
    
    if scaler is not None:
        query_feature_vec = scaler.transform([query_feature_vec])
    else:
        query_feature_vec = [query_feature_vec]
    labels, distances = hnsw_index.knn_query(query_feature_vec, k=topk)

    topk_res = []
    for i in range(len(query_feature_vec)):
        lbl_, dis_ = labels[i], distances[i]
        for j in range(len(lbl_)):
            topk_res.append(meta_data[lbl_[j]])
    common_res = [res for res in topk_res if res in jaccard_query_result]


    return jaccard_query_result, topk_res, common_res

def get_topk_hnsw_cols(lsh_index, hnsw_index, scaler, meta_data, query_df, intent_col_lst, intent_col_type, topk=20):
    query_feature_vec_lst = []

    for intent_col_id in intent_col_lst:
        if intent_col_type == 'num':
            query_values = [str(round(float(val_), 4)) for val_ in query_df.iloc[:, intent_col_id].dropna().values]
            query_feature_vec = convert_numerical_feature(clean_numerical_col(query_df.iloc[:, intent_col_id].astype(str)))
        elif intent_col_type == 'cate':
            query_values = genFunc.preprocessListValues(query_df.iloc[:, intent_col_id])
            query_feature_vec = extract_features(list(set(query_values)), query_df.columns[intent_col_id])

        if not check_for_invalid_values(query_feature_vec):
            query_feature_vec = handle_invalid_values(query_feature_vec)
            print('nan vectors!!')
        query_feature_vec_lst.append(query_feature_vec)

    
    # Transform the feature vectors
    if scaler is not None:
        query_feature_vec = scaler.transform(query_feature_vec_lst)
    else:
        query_feature_vec = query_feature_vec_lst

    # Perform HNSW query
    labels, distances = hnsw_index.knn_query(query_feature_vec, k=topk)
    
    topk_res = [[] for _ in range(len(query_feature_vec))]
    for i in range(len(query_feature_vec)):
        lbl_, dis_ = labels[i], distances[i]
        for j in range(len(lbl_)):
            topk_res[i].append(meta_data[lbl_[j]])


    return topk_res

def check_for_invalid_values(feature_vectors):
    if np.any(np.isinf(feature_vectors)):
        print("Feature vectors contain infinity values.")
        return False
    if np.any(np.isnan(feature_vectors)):
        print("Feature vectors contain NaN values.")
        return False
    if np.any(np.abs(feature_vectors) > 1e308):  # Check for extremely large values close to float64 limit
        print("Feature vectors contain extremely large values.")
        return False
    return True

def handle_invalid_values(feature_vec):
    if np.any(np.isnan(feature_vec)):
        feature_vec = np.nan_to_num(feature_vec, nan=0.0)
    if np.any(np.isinf(feature_vec)):
        feature_vec = np.nan_to_num(feature_vec, posinf=0.0, neginf=0.0)
    return feature_vec

def save_index_data(lsh_num_data, lsh_cate_data, num_meta_data, str_meta_data, num_feature_vectors, str_feature_vectors, num_save_index_path, cate_save_index_path, save_type='all'):
    if save_type=='num':
        num_containment_index = ColumnLSHEnsembleIndex()
        # print(lsh_num_data)

        for item in tqdm(lsh_num_data, total = len(lsh_num_data)):
            num_all_vals, tab_name, col_name, i = item[0], item[1], item[2], item[3]
            num_containment_index.add_num_index_data(num_all_vals, tab_name, col_name, i)

        num_lsh_index = num_containment_index.index()

        num_scaler = StandardScaler()
        num_feature_vectors = np.vstack(num_feature_vectors)
        num_feature_embeddings = num_scaler.fit_transform(num_feature_vectors)

        num_count_cols, num_dimensions = len(num_feature_vectors), len(num_feature_vectors[0])

        # Initialize the HNSW index
        num_hnsw_index = hnswlib.Index(space='l2', dim=num_dimensions)  # Use L2 (Euclidean) distance
        num_hnsw_index.init_index(max_elements=num_count_cols, ef_construction=200, M=16)

        # Add normalized feature vectors to the index
        num_hnsw_index.add_items(num_feature_embeddings)
        num_hnsw_index.set_ef(50)

        pickle.dump((num_lsh_index, num_hnsw_index, num_scaler, num_meta_data), open(num_save_index_path,'wb'))
    elif save_type=='cate':
        cate_containment_index = ColumnLSHEnsembleIndex()
        for item in tqdm(lsh_cate_data, total = len(lsh_cate_data)):
            unique_string_vals, tab_name, columnName, i = item[0], item[1], item[2], item[3]
            cate_containment_index.add_str_index_data(unique_string_vals, tab_name, columnName, i)

        cate_lsh_index = cate_containment_index.index()

        # cate_scaler = StandardScaler()
        # cate_feature_embeddings = cate_scaler.fit_transform(str_feature_vectors)
        cate_scaler = None
        cate_feature_embeddings = np.vstack(str_feature_vectors) 

        cate_count_cols, cate_dimensions = len(cate_feature_embeddings), len(cate_feature_embeddings[0])
        cate_hnsw_index = hnswlib.Index(space='l2', dim=cate_dimensions)  # Use L2 (Euclidean) distance
        cate_hnsw_index.init_index(max_elements=cate_count_cols, ef_construction=200, M=16)

        # Add normalized feature vectors to the index
        cate_hnsw_index.add_items(cate_feature_embeddings)
        cate_hnsw_index.set_ef(50)

        pickle.dump((cate_lsh_index, cate_hnsw_index, cate_scaler, str_meta_data), open(cate_save_index_path, 'wb'))

# python construct_index.py --datatype cate --benchmark $benchmark --groupindex $i --num_groups $num_groups
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process table data.")
    
    # Add arguments for benchmark and datatype
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--datatype', type=str, default='all', 
                        help="The data type to process. Must be one of 'num' or 'cate' or 'all'.")
    parser.add_argument('--groupindex', type=int, required=True,
                        help="In which the parallel it belongs to.")
    parser.add_argument('--num_groups', type=int, required=True,
                        help="The parallel settings for speed up the processing.")
    args = parser.parse_args()

    cur_benchmark = args.benchmark
    data_type = args.datatype
    iterate_group_index = args.groupindex

    # assert cur_benchmark in ['opendata', 'webtable', 'entitables', 'test']
    assert data_type in ['num', 'cate', 'all']
    # num_groups_dict = {'test':2, 'opendata':30, 'webtable': 30, 'entitables': 30}
    # num_groups = num_groups_dict[cur_benchmark]
    num_groups = args.num_groups

    assert iterate_group_index<num_groups

    datalake_path = os.path.abspath(f"../../{cur_benchmark}_benchmark/datalake/*.csv")
    
    total_files = glob.glob(datalake_path) 
    chunk_size = len(total_files) // num_groups

    if not len(total_files)>0:
        print('The data lake does not exist....')
        assert True
    if not os.path.exists(os.path.abspath('../test_hashmap')):
        os.mkdir(os.path.abspath('../test_hashmap'))

    start_idx = iterate_group_index * chunk_size
    end_idx = (iterate_group_index + 1) * chunk_size if iterate_group_index < num_groups - 1 else len(total_files)
    file_paths = total_files[start_idx:end_idx]


    if data_type =='num':
        num_meta_data, num_feature_vectors, lsh_num_data = [], [], []
        num_data_path = os.path.abspath(f"../test_hashmap/process_data_{cur_benchmark}_{iterate_group_index}_num.pkl")
    elif data_type == "cate":
        str_meta_data, str_feature_vectors, lsh_str_data = [], [], []
        cate_data_path = os.path.abspath(f"../test_hashmap/process_data_{cur_benchmark}_{iterate_group_index}_cate.pkl")

    if data_type == 'all':
        num_save_index_path = os.path.abspath(f"../test_hashmap/{cur_benchmark}_num_index.pkl")
        cate_save_index_path = os.path.abspath(f"../test_hashmap/{cur_benchmark}_cate_index.pkl")

        all_num_meta_data, all_num_feature_vectors, all_lsh_num_data = [], [], []
        all_cate_meta_data, all_cate_feature_vectors, all_lsh_cate_data = [], [], []
    
        print(f'begin to load numerical index data ... {num_save_index_path}')
        for i in tqdm(range(num_groups), total=num_groups):
            num_meta_data, num_feature_vectors, lsh_num_data = pickle.load(open(os.path.abspath(f"../test_hashmap/process_data_{cur_benchmark}_{i}_num.pkl"),'rb'))
            all_num_meta_data.extend(num_meta_data)
            all_num_feature_vectors.append(num_feature_vectors) 
            all_lsh_num_data.extend(lsh_num_data)
            save_index_data(all_lsh_num_data, all_lsh_cate_data, all_num_meta_data, all_cate_meta_data, all_num_feature_vectors, all_cate_feature_vectors,\
                num_save_index_path, cate_save_index_path, 'num')

        print(f'begin to load categorical index data ... {cate_save_index_path}')
        for i in tqdm(range(num_groups), total=num_groups):
            cate_meta_data, cate_feature_vectors, lsh_cate_data = pickle.load(open(os.path.abspath(f"../test_hashmap/process_data_{cur_benchmark}_{i}_cate.pkl"), "rb"))
            all_cate_meta_data.extend(cate_meta_data)
            all_cate_feature_vectors.append(cate_feature_vectors) 
            all_lsh_cate_data.extend(lsh_cate_data)

            save_index_data(all_lsh_num_data, all_lsh_cate_data, all_num_meta_data, all_cate_meta_data, all_num_feature_vectors, all_cate_feature_vectors,\
                num_save_index_path, cate_save_index_path, 'cate')
    else:
        # for f in tqdm(total_files, total = len(total_files)):
        for f in tqdm(file_paths, total = len(file_paths)):
            tab_name = f.split('/')[-1]

            try:
                if cur_benchmark in ['opendata']:
                    df = pd.read_csv(f, encoding='latin1', on_bad_lines="skip")
                else:
                    df = pd.read_csv(f )
            except:
                print(f"{f} error...")
                continue
            if len(df)>10000: df = df.sample(n = min(5000, int(len(df)*0.1)))
            if len(df)<=3:continue
            cur_cols = list(df.columns)
            for i in range(len(cur_cols)):
                columnName = cur_cols[i]

                if data_type=='cate':
                    if genFunc.getColumnType(df[columnName].tolist()) == 1: #check column Type
                        df[columnName] = df[columnName].map(str) 
                        string_vals = genFunc.preprocessListValues(df[columnName])
                        if len(list(set(string_vals)))<=1:continue
                        # print(columnName, 'cate')
                        str_meta_data.append(f"{tab_name} || {cur_cols[i]} || {i}")
                        feature_vec = extract_features(list(set(string_vals)), columnName)
                        if not check_for_invalid_values(feature_vec):
                            feature_vec = handle_invalid_values(feature_vec)
                            
                        str_feature_vectors.append(feature_vec)
                        unique_string_vals = set(string_vals)
                        lsh_str_data.append((unique_string_vals, tab_name, columnName, i))
                        # cate_containment_index.add_str_index_data(unique_string_vals, tab_name, columnName, i)
                if data_type=='num':
                    if is_numerical_col(df, cur_cols[i]):
                        num_vals = clean_numerical_col(df[cur_cols[i]].astype(str))
                        if len(num_vals.dropna())<=3:continue
                        num_meta_data.append(f"{tab_name} || {cur_cols[i]} || {i}")
                        feature_vec = convert_numerical_feature(num_vals)
                        if not check_for_invalid_values(feature_vec):
                            feature_vec = handle_invalid_values(feature_vec)

                        num_feature_vectors.append(feature_vec)
                        num_all_vals = [str(round(float(val_),4)) for val_ in df[cur_cols[i]].dropna().values]
                        lsh_num_data.append((num_all_vals, tab_name, cur_cols[i], i))

        if data_type == 'num':
            pickle.dump((num_meta_data, num_feature_vectors, lsh_num_data), open(num_data_path,'wb'))
        elif data_type == 'cate':
            pickle.dump((str_meta_data, str_feature_vectors, lsh_str_data), open(cate_data_path,'wb'))