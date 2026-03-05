import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, SelectKBest, chi2, f_classif
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from Datasets.get_dataset import get_dataset

def feature_selection_univariate(df, class_name, perc_num=50, perc_cat=50):
    """
    old function for feature selection, it is not used in the experiments
    :param df:
    :param class_name:
    :param perc_num:
    :param perc_cat:
    :return:
    """
    df_copy = df.copy()
    feature_cols = list(df_copy.columns)
    feature_cols.remove(class_name)
    X = df_copy[feature_cols]
    y = df_copy[class_name]
    categorical_columns = list(X.select_dtypes(include=['bool', 'object']).columns)
    numerical_columns = list(X.select_dtypes(include=['int64', 'float64']).columns)

    df_new = pd.DataFrame()
    num_cols_pre = len(numerical_columns)
    cat_cols_pre = len(categorical_columns)
    num_cols_post = 0
    cat_cols_post = 0

    if len(numerical_columns) != 0:
        #print("there are %d numerical columns" % len(numerical_columns))
        num_X = X[numerical_columns]
        if len(numerical_columns) > 1:
            selector = SelectPercentile(f_classif, percentile=perc_num)
            selector.fit(num_X, y)
            #idxs = np.argsort(selector.scores_)[::-1]
            #print("Mine, numerical:\n",num_X.columns[idxs])
            indices = np.argsort(selector.scores_[selector.get_support()])[::-1]
            num_features_names = selector.get_feature_names_out(num_X.columns)[indices]
        else:
            num_features_names = numerical_columns
            #print("Mine, numerical:\n", numerical_columns)
        df_new[num_features_names] = num_X[num_features_names]
        #print("selected %d numerical columns" % len(num_features_names))
        num_cols_post = len(num_features_names)

    if len(categorical_columns) != 0:
        #print("there are %d categorical columns" % len(categorical_columns))
        cat_X = X[categorical_columns]
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(cat_X)
        encoded_cat_X = oe.transform(cat_X)
        if len(categorical_columns) > 1:
            selector = SelectPercentile(chi2, percentile=perc_cat)
            selector.fit(encoded_cat_X, y)
            #idxs = np.argsort(selector.scores_)[::-1]
            #print("Mine, categorical:\n",cat_X.columns[idxs])
            indices = np.argsort(selector.scores_[selector.get_support()])[::-1]
            cat_feature_names = selector.get_feature_names_out(cat_X.columns)[indices]
        else:
            cat_feature_names = categorical_columns
            #print("Mine, categorical:\n", categorical_columns)
        df_new[cat_feature_names] = cat_X[cat_feature_names]
        #print("selected %d categorical columns" % len(cat_feature_names))
        cat_cols_post = len(cat_feature_names)

    df_new[class_name] = df_copy[class_name]
    return df_new, num_cols_pre, num_cols_post, cat_cols_pre, cat_cols_post

def remove_corr(df, class_name, threshold=0.8):
    """
    This function is used to remove features that are correlated more
    than the threshold
    :param df: full initial dataset
    :param class_name: name of the class column
    :param threshold: maximum allowed pearson correlation
    :return: the reduced dataset
    """
    # print("Features pre: ", len(df.columns)-1)
    df_copy = df.copy()
    feature_cols = list(df_copy.columns)
    feature_cols.remove(class_name)
    X = df_copy[feature_cols]
    corr = X.corr(numeric_only=True)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_copy.drop(df_copy[to_drop], axis=1, inplace=True)
    # print("Features post: ", len(df_copy.columns) - 1)
    return df_copy

def fixed_fs_univariate(df, class_name, cols_to_select=3):
    """
    This function performs a feature selection on the provided dataset.
    At most cols_to_select numerical columns are selected using the ANOVA test.
    At most cols_to_select categorical columns are selected using the Chi Square test.
    This is the function used in the Knowledge Base enrichment process for feature
    selection
    :param df: full dataset
    :param class_name: name of the class column
    :param cols_to_select: how many numerical and categorical columns to retain
    :return: the reduced dataset with only selected features
    """
    df_copy = df.copy()
    feature_cols = list(df_copy.columns)
    feature_cols.remove(class_name)
    X = df_copy[feature_cols]
    y = df_copy[class_name]
    categorical_columns = list(X.select_dtypes(include=['bool', 'object']).columns)
    numerical_columns = list(X.select_dtypes(include=['int64', 'float64', 'int32']).columns)
    df_new = pd.DataFrame()

    # select numerical columns
    if len(numerical_columns) != 0:
        num_X = X[numerical_columns]
        num_X_for_selection = num_X
        # SelectKBest/f_classif cannot handle NaNs; impute only for ranking.
        if num_X.isnull().values.any():
            num_X_for_selection = num_X_for_selection.fillna(num_X_for_selection.median())
            num_X_for_selection = num_X_for_selection.fillna(0.0)
        if len(numerical_columns) >= cols_to_select:
            selector = SelectKBest(f_classif, k=cols_to_select)
            selector.fit(num_X_for_selection, y)
            num_feature_names = selector.get_feature_names_out(num_X.columns)
        else:
            num_feature_names = numerical_columns
        df_new[num_feature_names] = num_X[num_feature_names]

    # select categorical columns
    if len(categorical_columns) != 0:
        cat_X = X[categorical_columns]
        cat_X_for_selection = cat_X
        # SelectKBest/chi2 cannot handle NaNs; use a placeholder only for ranking.
        if cat_X.isnull().values.any():
            cat_X_for_selection = cat_X.astype("object").fillna("__missing__")
        # Encode columns for correctly applying the statistical test.
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        oe.fit(cat_X_for_selection)
        encoded_cat_X = oe.transform(cat_X_for_selection)
        if len(categorical_columns) >= cols_to_select:
            selector = SelectKBest(chi2, k=cols_to_select)
            selector.fit(encoded_cat_X, y)
            cat_feature_names = selector.get_feature_names_out(cat_X.columns)
        else:
            cat_feature_names = categorical_columns
        df_new[cat_feature_names] = cat_X[cat_feature_names]

    df_new[class_name] = df_copy[class_name]
    return df_new
