import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_imp_methods_num = open("../Imputation/methods_numerical_column.txt", "r")
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_num = np.array([line.strip('\n\r') for line in imp_methods_num])
file_imp_methods_cat = open("../Imputation/methods_categorical_column.txt", "r")
imp_methods_cat = file_imp_methods_cat.readlines()
imp_methods_cat = np.array([line.strip('\n\r') for line in imp_methods_cat])

file_imp_methods_num.close()
file_imp_methods_cat.close()

def combine_num(thresh=0.005):
    """
    This function averages the raw experiments for numerical columns
    in the final files folder. It averages the f1 scores of the imputation
    methods and the profiles of the columns with the same missing value
    percentages. Finally, it saves for each "averaged" profile, only the
    equivalent imputation method according to a certain threshold.
    :param thresh: threshold for the computation of the equivalent imputation
    methods. An imputation method is equivalent to the best one if the difference
    in performance is no more than the provided threshold.
    """
    final_df = pd.read_csv("all_methods_exp/experiment_1_numerical.csv")
    num_columns = list(final_df.columns)
    num_columns.remove("name")
    num_columns.remove("column_name")
    num_columns.remove("ml_algorithm")
    for i in range(1,8):
        df = pd.read_csv(f"all_methods_exp/experiment_{i+1}_numerical.csv")
        final_df[num_columns] += df[num_columns]
    final_df[num_columns] /= 8

    # discover the best methods for each row
    scores = final_df[imp_methods_num].values
    sorted_scores_indices = np.argsort(scores, axis=1)[:, ::-1]
    sorted_scores = np.take_along_axis(scores, sorted_scores_indices, axis=1)
    best_methods = []

    # for each unit of knowledge, save the equivalent imputation methods
    for i in range(final_df.shape[0]):
        scores_row = sorted_scores[i, :]
        max_row = np.max(scores_row)
        diff_row = max_row - scores_row
        best_count = np.sum(diff_row <= thresh)
        best_methods.append(
            [str(m) for m in imp_methods_num[sorted_scores_indices[i, :]][:best_count]])
    count_list = np.array([len(x) for x in best_methods])
    vals, counts = np.unique(count_list, return_counts=True)

    plt.figure()
    plt.bar(vals, counts)
    plt.grid()
    plt.title("equivalent imputation methods for numerical columns")
    plt.xlabel("number of equivalent methods")
    plt.ylabel("number of column profiles")
    plt.ylim((0,1000))
    plt.xticks(range(1,10))
    plt.show()

    final_df["best_methods"] = best_methods

    # remove the numerical scores
    columns = list(final_df.columns)
    for col in imp_methods_num:
        columns.remove(col)

    # save the dataset
    dataset_to_save = final_df[columns]
    dataset_to_save.to_csv("combined_all/numerical_kb_combined.csv", index=False)

def combine_cat(thresh=0.005):
    """
    This function averages the raw experiments for categorical columns
    in the final files folder. It averages the f1 scores of the imputation
    methods and the profiles of the columns with the same missing value
    percentages. Finally, it saves for each "averaged" profile, only the
    equivalent imputation method according to a certain threshold.
    :param thresh: threshold for the computation of the equivalent imputation
    methods. An imputation method is equivalent to the best one if the difference
    in performance is no more than the provided threshold.
    """
    final_df = pd.read_csv("all_methods_exp/experiment_1_categorical.csv")
    num_columns = list(final_df.columns)
    num_columns.remove("name")
    num_columns.remove("column_name")
    num_columns.remove("ml_method")
    for i in range(1, 8):
        df = pd.read_csv(f"all_methods_exp/experiment_{i + 1}_categorical.csv")
        final_df[num_columns] += df[num_columns]
    final_df[num_columns] /= 8

    # discover the best methods for each row
    scores = final_df[imp_methods_cat].values
    sorted_scores_indices = np.argsort(scores, axis=1)[:, ::-1]
    sorted_scores = np.take_along_axis(scores, sorted_scores_indices, axis=1)
    best_methods = []
    for i in range(final_df.shape[0]):
        scores_row = sorted_scores[i, :]
        max_row = np.max(scores_row)
        diff_row = max_row - scores_row
        best_count = np.sum(diff_row <= thresh)
        best_methods.append(
            [str(m) for m in imp_methods_cat[sorted_scores_indices[i, :]][:best_count]])

    count_list = np.array([len(x) for x in best_methods])
    vals, counts = np.unique(count_list, return_counts=True)
    plt.figure()
    plt.bar(vals, counts)
    plt.grid()
    plt.title("equivalent imputation methods for categorical columns")
    plt.xlabel("number of equivalent methods")
    plt.ylabel("number of column profiles")
    plt.ylim((0, 1000))
    plt.show()

    final_df["best_methods"] = best_methods

    # remove the numerical scores
    columns = list(final_df.columns)
    for col in imp_methods_cat:
        columns.remove(col)

    dataset_to_save = final_df[columns]
    dataset_to_save.to_csv("combined_all/categorical_kb_combined.csv",index=False)


if __name__ == "__main__":
    combine_num()
    combine_cat()
