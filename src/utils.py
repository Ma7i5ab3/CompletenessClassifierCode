import pandas as pd
import numpy as np
import re

def encoding_categorical_variables(X):
    """
    Performs one hot encoding of the categorical variables in the provided
    dataset.
    :param X: dataset with not encoded categorical variables
    :return: the encoded dataset
    """
    def encode(original_dataframe, feature_to_encode):
        dummies = pd.get_dummies(original_dataframe[[feature_to_encode]], dummy_na=True)
        res = pd.concat([original_dataframe, dummies], axis=1)
        res = res.drop([feature_to_encode], axis=1)
        return res

    categorical_columns = list(X.select_dtypes(include=['bool','object']).columns)

    for col in X.columns:
        if col in categorical_columns:
            X = encode(X,col)
    return X

def restore_nans(df):
    nan_df = df.loc[:, df.columns.str.endswith("_nan")]

    pattern = "^([^_]*)_"
    regex = re.compile(pattern)

    for index in df.index:
        for col_nan in nan_df.columns:
            if df.loc[index,col_nan] == 1:
                col_id = regex.search(col_nan).group(1)
                targets = df.columns[df.columns.str.startswith(col_id+'_')]
                df.loc[index, targets] = np.nan
    
    df.drop(df.columns[df.columns.str.endswith('_nan')], axis=1, inplace=True)

    return df

def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return df

def dirty_single_column(dataset, column_name, name_class, seed):
    """
    Function that inject different percentages of missing values on a
    feature column.
    :param dataset: clean initial dataset
    :param column_name: name of the column to be injected with missing values
    :param name_class: name of the target class
    :param seed: seed for the random choice of positions of the missing values
    :return: a list of versions of the dataset, each with an increasing number
    of missing values in the indicated column
    """
    np.random.seed(seed)
    # il metodo usato è solo uniform
    df_pandas = dataset[[column_name]].copy()
    df_list = []

    perc = [0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05]
    for p in perc:
        df_dirt = df_pandas.copy()
        comp = [p, 1 - p]
        df_dirt = check_datatypes(df_dirt)
        for col in df_dirt.columns:

            if col != name_class:
                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)

                df_dirt.loc[rand == True, col] = np.nan

        # potrei fare qua il cambio da column a dataset
        df_dirt_complete = dataset.copy()
        df_dirt_complete[column_name] = df_dirt[column_name]
        column_type = df_dirt_complete[column_name].dtype
        print(f"Dirty Column: {column_name} Column type: {column_type}")
        df_list.append(df_dirt_complete)
        # print("saved {}-completeness{}%".format(column_name, round((1 - p) * 100)))
    return df_list