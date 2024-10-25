from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD, SoftImpute
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
import time
import argparse
import tqdm
from tqdm import tqdm
import time
from hyperimpute.plugins.imputers import Imputers
import json
import glob
from sklearn.preprocessing import MinMaxScaler
import random
random.seed(42)
import os
from types import SimpleNamespace
import sys

def re_convert_to_hivae_format(df_base, df_tgt):
    df_copy = df_tgt.copy()
    dict_uniques = dict()
    for col in df_base.columns:
        try:
            df_base[col] = df_base[col].astype(float)
        except:
        # if df_base[col].dtype == 'O':
            uniques = df_base[col].unique().tolist()
            dict_uniques[col] = {uniques[_i]: _i for _i in range(len(uniques))}
            
            # dict_uniques[col][np.nan] = ''
            # df_copy[col] = df_copy[col].apply(lambda x: dict_uniques[col][x])
            df_copy[col] = df_copy[col].apply(lambda x:np.nan if pd.isna(x) else dict_uniques[col].get(x,dict_uniques[col].get(str(x))))
            df_copy[col] = df_copy[col].replace({None:np.nan})

    return df_copy


def baseline_hyper_impute(X_incomplete, method='gain'):

    plugin = Imputers().get(method)
    out = plugin.fit_transform(X_incomplete.copy())

    return out


def baseline_inpute(X_incomplete, col_, X_complete=None, method='mean', dataset='opendata', level=0):
    time_cost = -1
    if method == 'mean':
        start_time = time.time()
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        end_time = time.time()
        return end_time-start_time, X_filled_mean
    elif method == 'knn':
        start_time = time.time()
        k = [3,10,50][level]
        X_filled_knn = KNN(k=k, verbose=False).fit_transform(X_incomplete)
        end_time = time.time()
        return end_time-start_time, X_filled_knn
    elif method == 'svd':
        start_time = time.time()
        min_max_scaler = MinMaxScaler()
        X_incomplete = min_max_scaler.fit_transform(X_incomplete)

        rank = [np.ceil((X_incomplete.shape[1]-1)/10),np.ceil((X_incomplete.shape[1]-1)/5),X_incomplete.shape[1]-1][level]
        X_filled_svd = IterativeSVD(rank=int(rank),verbose=False).fit_transform(X_incomplete)
        

        X_filled_svd = min_max_scaler.inverse_transform(X_filled_svd)

        end_time = time.time()
        return end_time-start_time, X_filled_svd
    elif method == 'normice':
        start_time = time.time()
        min_max_scaler = MinMaxScaler()
        X_incomplete = min_max_scaler.fit_transform(X_incomplete)

        max_iter = [3,10,50][level]
        X_filled_mice = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete)

        X_filled_mice = min_max_scaler.inverse_transform(X_filled_mice)

        end_time = time.time()
        return end_time-start_time, X_filled_mice
    elif method == 'mice':
        start_time = time.time()
        max_iter = [3,10,50][level]
        all_nan_columns = X_incomplete.columns[X_incomplete.isna().all()].tolist()
        if len(all_nan_columns)==0:
            X_filled_mice = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete)
            X_filled_mice = pd.DataFrame(X_filled_mice, columns=X_incomplete.columns)

        else:
            X_incomplete_non_nan = X_incomplete.drop(columns=all_nan_columns)
            X_filled_mf = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete_non_nan)

            X_filled_mf = pd.DataFrame(X_filled_mf, columns=X_incomplete_non_nan.columns, index=X_incomplete.index)
            X_filled_mf[all_nan_columns] = np.nan
            X_filled_mice = X_filled_mf[X_incomplete.columns]

        end_time = time.time()
        return end_time-start_time, X_filled_mice
    elif method == 'missforest':
        start_time = time.time()
        all_nan_columns = X_incomplete.columns[X_incomplete.isna().all()].tolist()
        if len(all_nan_columns)==0:
            plugin = Imputers().get("missforest")
            X_filled_mf = plugin.fit_transform(X_incomplete)
        else:
            X_incomplete_non_nan = X_incomplete.drop(columns=all_nan_columns)

            plugin = Imputers().get("missforest")
            X_filled_mf = plugin.fit_transform(X_incomplete_non_nan)

            X_filled_mf = pd.DataFrame(X_filled_mf, columns=X_incomplete_non_nan.columns, index=X_incomplete.index)

            X_filled_mf[all_nan_columns] = np.nan

            # Ensure the columns are in the original order of X_incomplete
            X_filled_mf = X_filled_mf[X_incomplete.columns]

        end_time = time.time()
        return end_time-start_time, X_filled_mf

    elif method == 'remice':
        start_time = time.time()
        max_iter = [3,10,50][level]
        imp_mean = IterativeImputer(max_iter=max_iter)
        imp_mean.fit(X_incomplete)
        X_filled_mice = X_incomplete.copy()
        for col in col_:
            col_index = X_incomplete.columns.get_loc(col)

            cur_incomplete_df = X_incomplete.copy()
            cur_incomplete_df[col] = np.NaN
            predict_col = imp_mean.transform(cur_incomplete_df)[:, col_index]
            X_filled_mice[col] = predict_col
        end_time = time.time()
        return end_time-start_time, X_filled_mice.values
    elif method == 'spectral':
        # default value for the sparsity level is with respect to the maximum singular value,
        # this is now done in a heuristic way
        start_time = time.time()
        sparsity = [0.5,None,3][level]
        X_filled_spectral = SoftImpute(shrinkage_value=sparsity).fit_transform(X_incomplete)
        end_time = time.time()
        return end_time-start_time, X_filled_spectral
    elif method == 'gain':
        start_time = time.time()
        # X_incomplete = X_incomplete.apply(pd.to_numeric, errors='coerce')
        X_incomplete = X_incomplete.astype(float)
        all_nan_columns = X_incomplete.columns[X_incomplete.isna().all()].tolist()
        if len(all_nan_columns)==0:
            output_df = baseline_hyper_impute(X_incomplete, method=method)
        else:
            X_incomplete_non_nan = X_incomplete.drop(columns=all_nan_columns)

            X_filled_mf = baseline_hyper_impute(X_incomplete_non_nan, method=method)

            # Convert the result back to a DataFrame with original column headers (excluding NaN columns)
            X_filled_mf = pd.DataFrame(X_filled_mf, columns=X_incomplete_non_nan.columns, index=X_incomplete.index)

            # Re-insert the previously dropped all-NaN columns in their original positions
            X_filled_mf[all_nan_columns] = np.nan

            # Ensure the columns are in the original order of X_incomplete
            output_df = X_filled_mf[X_incomplete.columns]

        end_time = time.time()
        return end_time - start_time, output_df.values
    
    elif method == 'regain':
        start_time = time.time()
        min_max_scaler = MinMaxScaler()
        X_incomplete = min_max_scaler.fit_transform(X_incomplete)
        # X_incomplete = X_incomplete.apply(pd.to_numeric, errors='coerce')
        X_incomplete = X_incomplete.astype(float)
        output_df = baseline_hyper_impute(X_incomplete, method='gain')
        output_df = min_max_scaler.inverse_transform(output_df)
        end_time = time.time()
        return end_time - start_time, output_df
    
    elif method == 'remiwae':
        start_time = time.time()
        min_max_scaler = MinMaxScaler()
        X_incomplete = min_max_scaler.fit_transform(X_incomplete)
        # X_incomplete = X_incomplete.apply(pd.to_numeric, errors='coerce')
        X_incomplete = X_incomplete.astype(float)
        output_df = baseline_hyper_impute(X_incomplete, method='miwae')
        output_df = min_max_scaler.inverse_transform(output_df)
        end_time = time.time()
        return end_time - start_time, output_df

        
    elif method == 'datawig':
        if X_incomplete[col_[0]].count()==0:
            return 0, X_incomplete
            
        import datawig
        from datawig import SimpleImputer, Imputer
        from datawig.column_encoders import CategoricalEncoder, NumericalEncoder
        from datawig.mxnet_input_symbols import EmbeddingFeaturizer, NumericalFeaturizer
        print(f'running method {method}...')

        all_nan_columns = X_incomplete.columns[X_incomplete.isna().all()].tolist()
        if len(all_nan_columns)>0:
            X_incomplete_non_nan = X_incomplete.drop(columns=all_nan_columns)
            dirty_df = X_incomplete_non_nan.copy()
        else:
            dirty_df = X_incomplete.copy()
        
        data_cols = dirty_df.columns
        impute_time_cost = 0
        # print(data_cols)

        data_encoder_dict, data_featurizer_dict = {}, {}
        for col in data_cols:
            if dirty_df[col].dtype==object:
                # if len(dirty_df[col].unique())<100:
                data_encoder_dict[col] = CategoricalEncoder(col)
                data_featurizer_dict[col] = EmbeddingFeaturizer(col)
            else:
                data_encoder_dict[col] = NumericalEncoder(col)
                data_featurizer_dict[col] = NumericalFeaturizer(col)
                dirty_df[col] = dirty_df[col].astype(float)

        for col in col_:
            cnt = len(dirty_df[~dirty_df[col].isnull()])
            train_df =  dirty_df[~dirty_df[col].isnull()]
            # dirty_df[~dirty_df[col].isnull()].sample(n=min(5000, cnt))
            if len(train_df)<10:
                train_df = dirty_df[~dirty_df[col].isnull()]
                print('error!!! less than 10 rows, we adjust the rows as ', len(train_df))

            data_encoder_cols = [data_encoder_dict[c] for c in data_cols if c!=col and c in data_encoder_dict]
            label_encoder_cols = [data_encoder_dict[col]]
            data_featurizer_cols = [data_featurizer_dict[c] for c in data_cols if c!=col and c in data_featurizer_dict]

            imputer = Imputer(
                    data_featurizers = data_featurizer_cols,
                    data_encoders = data_encoder_cols,
                    label_encoders = label_encoder_cols,
                    output_path = f'db_test_datawig/dg_imputer_model_'+col
            )

            try:
                print('begin to train ...')
                imputer.fit(train_df = train_df)
                start_time = time.time()
                prediction_df = imputer.predict(dirty_df)[col+'_imputed']
                X_incomplete[col] = prediction_df
                end_time = time.time()

                impute_time_cost += end_time - start_time
                
            except:
                print(col, ' meet with some errors')
        return impute_time_cost, X_incomplete