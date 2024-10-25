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
import tree_influence
from tree_influence.explainers import BoostIn
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from collections import Counter

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

def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


class MissForest(BaseEstimator, TransformerMixin):
    """Missing value imputation using Random Forests.

    MissForest imputes missing values using Random Forests in an iterative
    fashion. By default, the imputer begins imputing missing values of the
    column (which is expected to be a variable) with the smallest number of
    missing values -- let's call this the candidate column.
    The first step involves filling any missing values of the remaining,
    non-candidate, columns with an initial guess, which is the column mean for
    columns representing numerical variables and the column mode for columns
    representing categorical variables. After that, the imputer fits a random
    forest model with the candidate column as the outcome variable and the
    remaining columns as the predictors over all rows where the candidate
    column values are not missing.
    After the fit, the missing rows of the candidate column are
    imputed using the prediction from the fitted Random Forest. The
    rows of the non-candidate columns act as the input data for the fitted
    model.
    Following this, the imputer moves on to the next candidate column with the
    second smallest number of missing values from among the non-candidate
    columns in the first round. The process repeats itself for each column
    with a missing value, possibly over multiple iterations or epochs for
    each column, until the stopping criterion is met.
    The stopping criterion is governed by the "difference" between the imputed
    arrays over successive iterations. For numerical variables (num_vars_),
    the difference is defined as follows:

     sum((X_new[:, num_vars_] - X_old[:, num_vars_]) ** 2) /
     sum((X_new[:, num_vars_]) ** 2)

    For categorical variables(cat_vars_), the difference is defined as follows:

    sum(X_new[:, cat_vars_] != X_old[:, cat_vars_])) / n_cat_missing

    where X_new is the newly imputed array, X_old is the array imputed in the
    previous round, n_cat_missing is the total number of categorical
    values that are missing, and the sum() is performed both across rows
    and columns. Following [1], the stopping criterion is considered to have
    been met when difference between X_new and X_old increases for the first
    time for both types of variables (if available).

    Parameters
    ----------
    NOTE: Most parameter definitions below are taken verbatim from the
    Scikit-Learn documentation at [2] and [3].

    valid_misscount_idx: I want to specify the dimension of missing features

    max_iter : int, optional (default = 10)
        The maximum iterations of the imputation process. Each column with a
        missing value is imputed exactly once in a given iteration.

    decreasing : boolean, optional (default = False)
        If set to True, columns are sorted according to decreasing number of
        missing values. In other words, imputation will move from imputing
        columns with the largest number of missing values to columns with
        fewest number of missing values.

    missing_values : np.nan, integer, optional (default = np.nan)
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    criterion : tuple, optional (default = ('mse', 'gini'))
        The function to measure the quality of a split.The first element of
        the tuple is for the Random Forest Regressor (for imputing numerical
        variables) while the second element is for the Random Forest
        Classifier (for imputing categorical variables).

    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    class_weight : dict, list of dicts, "balanced", "balanced_subsample" or \
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
        NOTE: This parameter is only applicable for Random Forest Classifier
        objects (i.e., for categorical variables).

    Attributes
    ----------
    statistics_ : Dictionary of length two
        The first element is an array with the mean of each numerical feature
        being imputed while the second element is an array of modes of
        categorical features being imputed (if available, otherwise it
        will be None).

    """

    def __init__(self, valid_misscount_idx, max_iter=10, decreasing=False, missing_values=np.nan,
                 copy=True, n_estimators=100, criterion=('mse', 'gini'),
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto',
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
                 verbose=0, warm_start=False, class_weight=None):

        self.valid_misscount_idx = valid_misscount_idx
        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

    def _miss_forest(self, Ximp, mask):
        """The missForest algorithm"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)

            # Reg criterion
            reg_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[0]

            rf_regressor = LGBMRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs)

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

            # Classfication criterion
            clf_criterion = self.criterion if type(self.criterion) == str \
                else self.criterion[1]

            rf_classifier = LGBMClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        misscount_idx = [_ for _ in misscount_idx if _ in self.valid_misscount_idx]
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf
        gamma_newcat = 0
        gamma_oldcat = np.inf
        col_index = np.arange(Ximp.shape[1])

        saved_model_dict = {}

        while (
                gamma_new < gamma_old or gamma_newcat < gamma_oldcat) and \
                self.iter_count_ < self.max_iter:

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new
                gamma_oldcat = gamma_newcat
            # 5. loop
            for s in misscount_idx:
                # Column indices other than the one being imputed
                s_prime = np.delete(col_index, s)

                # Get indices of rows where 's' is observed and missing
                obs_rows = np.where(~mask[:, s])[0]
                mis_rows = np.where(mask[:, s])[0]

                # If no missing, then skip
                if len(mis_rows) == 0:
                    continue

                # Get observed values of 's'
                yobs = Ximp[obs_rows, s]

                # Get 'X' for both observed and missing 's' column
                xobs = Ximp[np.ix_(obs_rows, s_prime)]
                xmis = Ximp[np.ix_(mis_rows, s_prime)]

                # 6. Fit a random forest over observed and predict the missing
                if self.cat_vars_ is not None and s in self.cat_vars_:
                    rf_classifier.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_classifier.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

                    saved_model_dict[s] = rf_classifier
                else:
                    rf_regressor.fit(X=xobs, y=yobs)
                    # 7. predict ymis(s) using xmis(x)
                    ymis = rf_regressor.predict(xmis)
                    # 8. update imputed matrix using predicted matrix ymis(s)
                    Ximp[mis_rows, s] = ymis

                    saved_model_dict[s] = rf_regressor

            # 9. Update gamma (stopping criterion)
            if self.cat_vars_ is not None:
                gamma_newcat = np.sum(
                    (Ximp[:, self.cat_vars_] != Ximp_old[:, self.cat_vars_])) / n_catmissing
            if self.num_vars_ is not None:
                gamma_new = np.sum((Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_]) ** 2) / np.sum((Ximp[:, self.num_vars_]) ** 2)

            print("Iteration:", self.iter_count_)
            self.iter_count_ += 1


        return Ximp_old, saved_model_dict

    def process_data_fit(self, Ximp, mask):
        """Data processing"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')
            Ximp[missing_num_rows, missing_num_cols] = np.take(
                col_means, missing_num_cols)

        # If needed, repeat for categorical variables
        if self.cat_vars_ is not None:
            # Calculate total number of missing categorical values (used later)
            n_catmissing = np.sum(mask[:, self.cat_vars_])

            # Only keep indices for categorical vars
            keep_idx_cat = np.in1d(missing_cols, self.cat_vars_)
            missing_cat_rows = missing_rows[keep_idx_cat]
            missing_cat_cols = missing_cols[keep_idx_cat]

            # Make initial guess for missing values
            col_modes = np.full(Ximp.shape[1], fill_value=np.nan)
            col_modes[self.cat_vars_] = self.statistics_.get('col_modes')
            Ximp[missing_cat_rows, missing_cat_cols] = np.take(col_modes, missing_cat_cols)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        misscount_idx = [_ for _ in misscount_idx if _ in self.valid_misscount_idx]
        
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]
        if len(misscount_idx)==0:
            misscount_idx = self.valid_misscount_idx

        col_index = np.arange(Ximp.shape[1])

        save_data_dict = {}
        # 5. loop
        for s in misscount_idx:
            # Column indices other than the one being imputed
            s_prime = np.delete(col_index, s)

            # Get indices of rows where 's' is observed and missing
            obs_rows = np.where(~mask[:, s])[0]
            mis_rows = np.where(mask[:, s])[0]

            # If no missing, then skip
            # if len(mis_rows) == 0:
            #     continue

            # Get observed values of 's'
            yobs = Ximp[obs_rows, s]

            # Get 'X' for both observed and missing 's' column
            xobs = Ximp[np.ix_(obs_rows, s_prime)]
            
            save_data_dict[s] = (xobs, yobs)
        # print('---1', save_data_dict)

        return save_data_dict

    def fit(self, X, y=None, cat_vars=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        cat_vars : int or array of ints, optional (default = None)
            An int or an array containing column indices of categorical
            variable(s)/feature(s) present in the dataset X.
            ``None`` if there are no categorical variables in the dataset.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True

        self.label_encoders_ = {}
        if cat_vars is not None:
            for cat_var in cat_vars:
                le = LabelEncoder()
                X[:, cat_var] = le.fit_transform(pd.Series(X[:, cat_var]).astype(str).fillna('missing'))
                self.label_encoders_[cat_var] = le


        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            raise ValueError("One or more columns have all rows missing.")

        if cat_vars is not None:
            if type(cat_vars) == int:
                cat_vars = [cat_vars]
            elif type(cat_vars) == list or type(cat_vars) == np.ndarray:
                if np.array(cat_vars).dtype != int:
                    raise ValueError(
                        "cat_vars needs to be either an int or an array "
                        "of ints.")
            else:
                raise ValueError("cat_vars needs to be either an int or an array "
                                 "of ints.")

        # Identify numerical variables
        num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
        num_vars = num_vars if len(num_vars) > 0 else None

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = np.nanmean(X[:, num_vars], axis=0) if num_vars is not None else None
        col_modes = mode(
            X[:, cat_vars], axis=0, nan_policy='omit')[0] if cat_vars is not \
                                                           None else None

        self.cat_vars_ = cat_vars
        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means, "col_modes": col_modes}

        return self

    def transform(self, X, mode='train'):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """
        # Confirm whether fit() has been called
        check_is_fitted(self, ["cat_vars_", "num_vars_", "statistics_"])

        # Check data integrity
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True
        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            return None
            # raise ValueError("One or more columns have all rows missing.")

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (0 if self.num_vars_ is None else len(self.num_vars_)) \
            + (0 if self.cat_vars_ is None else len(self.cat_vars_))
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")

        # Check if anything is actually missing and if not return original X                             
        mask = _get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original "
                          "dataset.")
            return None
            # return X

        # row_total_missing = mask.sum(axis=1)
        # if not np.any(row_total_missing):
        #     return X
        if mode == 'train':
            X, predict_model_dict = self._miss_forest(X, mask)
            return X, predict_model_dict
        elif mode=='eval':
            return self.process_data_fit(X, mask)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X,mode='train')


def construct_clusters(input_x):
    """
    Constructs clusters for the given input data and returns a dictionary of sample indices and their corresponding cluster labels.

    Args:
        input_x (array-like): The input data to cluster.

    Returns:
        dict: A dictionary with indices as keys and cluster labels as values.
    """
    label_dict = {}

    # Scale the input data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_x)

    # Apply KMeans clustering
    # kmeans = KMeans(n_clusters=50, random_state=42)  # Set the number of clusters as needed
    kmeans = MiniBatchKMeans(n_clusters=50, batch_size=10000, random_state=42)
    kmeans.fit(scaled_data)

    # Map each sample index to its cluster label
    for i in range(len(input_x)):
        label_dict[i] = int(kmeans.labels_[i])

    return label_dict


def mask_values_no_index(dataset, col_idx, mask_fraction=0.2):
    # Get the number of rows to mask (20% of the column)
    num_rows_to_mask = int(len(dataset) * mask_fraction)

    # Randomly select row indices (using positional indices instead of dataset.index)
    mask_indices = np.random.choice(len(dataset), size=num_rows_to_mask, replace=False)

    # Set the selected positional indices in the specified column to NaN
    dataset.iloc[mask_indices, col_idx] = np.nan

    return dataset

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def identify_useful_rows(input_test_df, bound_, missing_col_idx, threshold = 1000):
    test_df = input_test_df.copy()
    current_string_cols = []
    for i in range(test_df.shape[1]):
        try:
            test_df.iloc[:, i] = test_df.iloc[:, i].astype(float)
        except:
            current_string_cols.append(i)
   
    encode_dict = {}

    for col in current_string_cols:
        # Get the unique values, excluding NaN
        unique_vals = test_df.iloc[:, col].dropna().unique()
        
        # Create a dictionary to map each unique value to an integer index
        unique_val_dict = {unique_vals[i]: i for i in range(len(unique_vals))}
        
        # Apply the dictionary to encode the column, handling NaN values separately
        if col!=missing_col_idx:
            test_df.iloc[:, col] = test_df.iloc[:, col].apply(lambda x: unique_val_dict.get(x, np.nan))
        
        # Store the encoding dictionary for future reference
        encode_dict[col] = unique_val_dict

    # test_df filter using statistics at first
    try:
        test_df.iloc[:bound_, missing_col_idx] = test_df.iloc[:bound_, missing_col_idx].astype(float)
        col_dtype = 'num'
        min_val, max_val = test_df.iloc[:bound_, missing_col_idx].min(), test_df.iloc[:bound_, missing_col_idx].max()
        external_tbl2 = test_df.iloc[bound_:, ].copy()
        external_tbl2 = external_tbl2[external_tbl2.iloc[:, missing_col_idx].apply(is_numeric)]
        external_tbl2.iloc[:, missing_col_idx] = external_tbl2.iloc[:, missing_col_idx].astype(float)
        external_dataset = external_tbl2[(external_tbl2.iloc[:, missing_col_idx] >= min_val) & 
                    (external_tbl2.iloc[:, missing_col_idx] <= max_val)]
        
        if  missing_col_idx in encode_dict:
            encode_dict = {key:val for key, val in encode_dict.items() if key!=missing_col_idx}

    except:
        # print(encode_dict[missing_col_idx])
        test_df.iloc[:, missing_col_idx] = test_df.iloc[:, missing_col_idx].apply(lambda x: encode_dict[missing_col_idx].get(x, np.nan))

        common_vals = test_df.iloc[:bound_, missing_col_idx].unique().tolist()
        external_tbl2 = test_df.iloc[bound_:, ].copy()
        external_dataset = external_tbl2[external_tbl2.iloc[:, missing_col_idx].isin(common_vals)]

    test_df = pd.concat([test_df.iloc[:bound_], external_dataset], axis=0)
    print(f'before dataset size {len(input_test_df)}, after dataset size {len(test_df)}')


    # Ensure the dataset size is capped at 500
    if bound_ > 500:
        train_valid_dataset = test_df[:bound_].sample(n=500, random_state=42)
    else:
        train_valid_dataset = test_df[:bound_]

    train_valid_dataset = mask_values_no_index(train_valid_dataset, missing_col_idx, mask_fraction=0.2)

    row_num, col_num = train_valid_dataset.shape
    valid_dims_ = []
    for i_col in range(col_num):
        # print(train_valid_dataset.iloc[:, col].count()/len(train_valid_dataset))
        if train_valid_dataset.iloc[:, i_col].count()/len(train_valid_dataset)>0.2:
            valid_dims_.append(i_col)
    train_valid_dataset = train_valid_dataset.iloc[:, valid_dims_]  
    try:
        valid_missing_col_idx = valid_dims_.index(missing_col_idx)
    except:
        return {}

    imputer = MissForest([valid_missing_col_idx], random_state=1337)
    # print('---', valid_missing_col_idx)
    train_data, test_data = train_test_split(train_valid_dataset, test_size=0.2, random_state=42)
    valid_data = test_df[bound_:].iloc[:, valid_dims_]
    # in case valid_data non missing values
    # valid_data sample 1 row of valid_missing_col_idx to be nan 
    if len(valid_data)==0:
        return {}
    valid_data.iloc[0, valid_missing_col_idx] = np.nan

    train_notnull_idx = train_valid_dataset[train_valid_dataset.iloc[:, valid_missing_col_idx].notnull()]
    split_dim = train_notnull_idx.sample(1)
    train_data = pd.concat([split_dim, train_data], axis=0)
    test_data = pd.concat([split_dim, test_data], axis=0)

    print('ready to fit...')
    basic_columns = valid_data.columns
    dim_valid_sample_dict = {}
    output_res, pretrained_model = imputer.fit_transform(train_data.values)
    test_data_dict = imputer.transform(test_data.values, mode='eval' )
    valid_data_dict = imputer.transform(valid_data.values, mode='eval' )

    if not valid_data_dict or not test_data_dict:
        return {}

    
    for dim_, model_ in pretrained_model.items():
        X_valid, y_valid = valid_data_dict[dim_]
        current_basic_columns = [basic_columns[_] for _ in range(len(basic_columns)) if _!=dim_]

        X_test, y_test = test_data_dict[dim_]
        
        if len(X_valid) > threshold:
            cluster_dict = construct_clusters(X_valid)

            # Randomly sample 1000 items and their indices from X_test and y_test
            sample_indices = random.sample(range(len(X_valid)), 1000)
            sample_X_valid = [X_valid[i] for i in sample_indices]
            sample_y_valid = [y_valid[i] for i in sample_indices]

            explainer = BoostIn().fit(model_, np.array(sample_X_valid), np.array(sample_y_valid))
            influence = explainer.get_local_influence(X_test, y_test)

            sample_valid_keys = [cluster_dict[sample_indices[i]] for i, values in enumerate(influence) if sum(values) > 0]

            # Find the top-3 most frequent cluster ids
            top3_clusterids = [cluster_id for cluster_id, _ in Counter(sample_valid_keys).most_common(3)]

            # Filter samples belonging to the top-3 cluster IDs
            incluster_samples = [i for i in range(len(X_valid)) if cluster_dict[i] in top3_clusterids]
            if len(incluster_samples)==0:
                valid_keys = []
            else:
                explainer = BoostIn().fit(model_, np.array([X_valid[i] for i in incluster_samples]), np.array([y_valid[i] for i in incluster_samples]))
                # Calculate influence for filtered samples in the top clusters
                influence = explainer.get_local_influence(X_test, y_test)

                # Select valid keys based on positive influence
                valid_keys = [incluster_samples[i] for i, values in enumerate(influence) if sum(values) > 0]
                # print('time for final selection is ', time.time() - start_time, len(valid_keys))

        else:
            print(f'due to {threshold}, we change it as no sampling')
            # Calculate local influence when the dataset is small
            explainer = BoostIn().fit(model_, X_valid, y_valid)
            influence = explainer.get_local_influence(X_test, y_test)

            valid_keys = []
            for i, values in enumerate(influence):
                if sum(values) > 0:
                    valid_keys.append(i)

        feature_importances = model_.feature_importances_
        
        valid_feature_dims = filter_feature_importances(feature_importances, cumulative_threshold=0.90)
        valid_feature_colnames = [current_basic_columns[_] for _ in valid_feature_dims.tolist()]

        dim_valid_sample_dict[dim_] = [test_df, bound_, valid_keys, valid_feature_colnames, basic_columns[dim_], encode_dict]

    return dim_valid_sample_dict


def filter_feature_importances(feature_importances, cumulative_threshold=0.90):
    # Sort feature importances in descending order
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_indices]

    # Calculate cumulative importance
    cumulative_importance = np.cumsum(sorted_importances) / np.sum(sorted_importances)
    # Find the top-k features that make up the cumulative importance threshold
    k = np.argmax(cumulative_importance >= cumulative_threshold) + 1  # Add 1 since np.argmax returns the first index

    # Get the indices of the top-k important features
    top_k_indices = sorted_indices[:k]

    return top_k_indices

def process_query_table(query_table_col_df, benchmark_path, candidate_map_res):
    datalake_tabs = {}
    datalake_tabs_header = {}
    
    potential_tabs = {key for related_infos in candidate_map_res.values() for key in related_infos.keys()}

    for pt_tab in tqdm(potential_tabs, total=len(potential_tabs)):
        pt_tab_path = os.path.join(benchmark_path, 'datalake', pt_tab)
        datalake_tabs[pt_tab] = pd.read_csv(pt_tab_path, encoding='latin1')
        datalake_tabs_header[pt_tab] = datalake_tabs[pt_tab].columns.tolist()

    print(f'{len(datalake_tabs)} unionable tables loading done...')

    all_search_time = 0
    all_cnt = 0

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
        if query_long_index not in candidate_map_res:continue

        start_time = time.time()

        items = list(candidate_map_res[query_long_index].items())

        if len(items) > 50:
            items = random.sample(items, 50)  # Randomly sample 50 items
            print(f'too many of {len(items)}')
            # candidate_map_res[query_long_index].items()

        for candidate_name, query_candidate_map_cols in items:
            if len(query_candidate_map_cols) == 0:
                continue

            map_id, query_cols, candidate_cols = query_candidate_map_cols
            
            if len(query_cols) == 0:
                continue

            compare_cols = set([_.lower() for _ in query_cols]) & set([_.lower() for _ in candidate_cols])
            if len(compare_cols)==0:continue


            query_cols = get_columns(query_cols, input_table.columns.tolist())
            missing_column = get_single_column(row['column_name'], input_table.columns.tolist())
            missing_column_flag = missing_column in query_cols
            # assert missing_column not in query_cols

            # Extract the query tuple (example: 2nd row)
            if not missing_column_flag:
                query_cols.append(missing_column)

            raw_query_table = input_table[query_cols]
            candidate_col_indices = []
            for ii in range(len(candidate_cols)):
                col_ = candidate_cols[ii]
                try:
                    candidate_col_indices.append(datalake_tabs_header[candidate_name].index(col_))
                except:
                    close_matches = difflib.get_close_matches(col_, datalake_tabs_header[candidate_name], n=1)
                    if len(close_matches) > 0:
                        candidate_col_indices.append(datalake_tabs_header[candidate_name].index(close_matches[0]))
                    else:
                        del query_cols[ii]
                        # print(col_, datalake_tabs_header[candidate_name])
                        # break
                        continue

            if len(candidate_col_indices) == 0:
                print(f"missing {candidate_cols}")
                continue
            # Use the whole table as candidate tuples (excluding the query tuple)
            if not missing_column_flag:
                candidate_col_indices.append(map_id)
            candidate_df = datalake_tabs[candidate_name].iloc[:,candidate_col_indices]

            # left_on_columns = input_table.columns[query_cols].tolist()
            left_on_columns = query_cols
            right_on_columns = candidate_df.columns

            assert len(left_on_columns) == len(right_on_columns)
             
            updated_candidate_df = pd.DataFrame(columns=input_table.columns)
            for left_col, right_col in zip(left_on_columns, right_on_columns):
                if left_col in input_table.columns and right_col in candidate_df.columns:
                    # print(left_col, right_col, candidate_name, datalake_tabs[candidate_name].columns)
                    # print(candidate_df.columns)
                    # print(candidate_df.columns, candidate_df.columns.duplicated().any())
                    duplicated_cols = candidate_df.columns == right_col

                    # Check if any duplicated columns exist
                    if duplicated_cols.sum() > 1:
                        # Get the actual indices of the duplicated columns (where the mask is True)
                        duplicated_indices = np.where(duplicated_cols)[0]
                        # print(f"Multiple occurrences found for '{right_col}' at indices {duplicated_indices}. Taking the first occurrence.")
                        updated_candidate_df[left_col] = candidate_df.iloc[:, duplicated_indices[0]].values
                    else:
                        # If it's not duplicated, simply use the column directly
                        updated_candidate_df[left_col] = candidate_df[right_col].values

            input_table = pd.concat([input_table, updated_candidate_df], axis=0)
            input_table = input_table.fillna(np.nan)

        valid_table = input_table[bound_:].copy()

        if len(valid_table)>0:
            for col_ in valid_table.columns:
                if valid_table[col_].count() == 0:
                    input_table = input_table.drop(columns=[col_])
                    
            merged_table_dict[query_long_index] = [input_table, bound_]
        else:
            print('no external data...')

        end_time = time.time()
        all_search_time += end_time - start_time

        print(f"Processing {all_cnt + 1} of {len(query_table_col_df)} done..." )
        all_cnt += 1

    print(f"Total search time is {all_search_time}")
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process table data.")
    # Add arguments for benchmark and datatype
    parser.add_argument('--benchmark', type=str, required=True, 
                        help="The benchmark to use. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    parser.add_argument('--raw_benchmark', type=str, required=True, 
                        help="The benchmark that is used to find the relevant tables. Must be one of 'opendata', 'webtable', 'entitables', 'test'.")
    args = parser.parse_args()
    current_benchmark = args.benchmark
    raw_benchmark = args.raw_benchmark

    threshold = 5000
    benchmark_path = os.path.abspath(f'../../{current_benchmark}_benchmark/')
    query_table_col_df = pd.read_csv(f'{benchmark_path}/missing_query/missing_tab_row_col.csv')

    tmp_path = os.path.abspath("./tmp_files")
    os.makedirs(tmp_path, exist_ok=True)

    file_path = os.path.abspath(f"../test_stats/{raw_benchmark}_benchmark_result_by_santos_estimate.pkl")

    with bz2.BZ2File(file_path, "rb") as file:
        candidate_map_res = pickle.load(file)
 
    
    print('Begin processing...')
    if not os.path.exists(os.path.join(tmp_path, f'{current_benchmark}.pkl')):
        merged_table_dict, table_column_row_idx_dict, query_tab_gt_val_dict = process_query_table(query_table_col_df, benchmark_path, candidate_map_res)
        pickle.dump((merged_table_dict, table_column_row_idx_dict, query_tab_gt_val_dict), open(os.path.join(tmp_path, f'{current_benchmark}.pkl'),\
            'wb'))
    else:
        merged_table_dict, table_column_row_idx_dict, query_tab_gt_val_dict = pickle.load(open(os.path.join(tmp_path, f'{current_benchmark}.pkl'),\
            'rb'))
    extended_output_row_dict, = {}, 
    total_time_cost1, total_time_cost2 = 0, 0
    error_impute = 0

    sampled_items = list(merged_table_dict.items())

    # Iterate over the sampled items
    for query_long_index, merged_candidate_infos in tqdm(sampled_items, total=len(sampled_items)):
        merged_data, raw_dataset_size,  = merged_candidate_infos


        missing_col_name = query_long_index.split(' || ')[-2]
        
        try:
            missing_col_idx = merged_data.columns.get_loc(missing_col_name)
        except:
            missing_col_name = get_single_column(missing_col_name, merged_data.columns)
            missing_col_idx = merged_data.columns.tolist().index(missing_col_name)

        start_time = time.time()
        filter_row_feat_dict = identify_useful_rows(merged_data.copy(), raw_dataset_size, missing_col_idx, threshold)
        enhance_time_cost = time.time() - start_time
        print('per enhance time cost: ', enhance_time_cost)
        total_time_cost2 += enhance_time_cost
        flag = True
        
        try:
            for col_, unioned_infos in filter_row_feat_dict.items():
                print(col_, query_long_index)
                
                big_table, bound, valid_rows, _, missing_col, cate_encode_dict = unioned_infos
                big_table_complete = big_table.copy()

                try:
                    big_table.iloc[:bound, missing_col_idx] = big_table.iloc[:bound, missing_col_idx].astype(float)
                    col_dtype = 'num'
                    min_val, max_val = big_table.iloc[:bound, missing_col_idx].min(), big_table.iloc[:bound, missing_col_idx].max()
                except:
                    common_vals = big_table.iloc[:bound, missing_col_idx].unique().tolist()
                    col_dtype = 'cate'

                for cur_row_idx in table_column_row_idx_dict[query_long_index]:
                    # print(query_long_index, cur_row_idx, query_tab_gt_val_dict[query_long_index])
                    big_table_complete.iloc[cur_row_idx, col_] = query_tab_gt_val_dict[query_long_index][cur_row_idx]

                if col_ in cate_encode_dict:
                    raw_key_val_dict = {val:key for key, val in cate_encode_dict[col_].items()}
                else:
                    raw_key_val_dict = None

                external_tbl2 = big_table.iloc[valid_rows].copy()
                if col_dtype == 'num':  
                    external_tbl2 = external_tbl2[(external_tbl2.iloc[:, missing_col_idx] >= min_val) & 
                                (external_tbl2.iloc[:, missing_col_idx] <= max_val)]
                else:
                    external_tbl2 = external_tbl2[external_tbl2.iloc[:, missing_col_idx].isin(common_vals)]
                updated_table = pd.concat([big_table[:bound], external_tbl2], axis=0)

                extended_output_row_dict[query_long_index] = updated_table

        except:
            flag = False
            error_impute += 1
            print('error when imputing')
        if len(filter_row_feat_dict)==0 or not flag:
            print(f'lack of training data... due to lacking of external training data? {len(filter_row_feat_dict)==0}')

    print(f'{current_benchmark} query done... with {error_impute} columns')
    print('row selection time cost: ', total_time_cost2)

    pickle.dump(extended_output_row_dict,\
        open(os.path.join(tmp_path, f'{current_benchmark}_enhance.pkl'),'wb'))