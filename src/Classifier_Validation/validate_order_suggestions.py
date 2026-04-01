import numpy as np
import matplotlib.pyplot as plt
from Feature_selection.feature_selection import fixed_fs_univariate
from Imputation.imputation_techniques import impute_missing_column
from classifiers_validation import try_classification
from Column_profile_extraction.numerical import get_features_num
from Column_profile_extraction.categorical import get_features_cat
from joblib import load
from Datasets.get_dataset import get_dataset
import itertools
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

file_imp_methods_num = open("Imputation/methods_numerical_column.txt", "r")
file_imp_methods_cat = open("Imputation/methods_categorical_column.txt", "r")
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_cat = file_imp_methods_cat.readlines()
imp_methods_num = [line.strip('\n\r') for line in imp_methods_num]
imp_methods_cat = [line.strip('\n\r') for line in imp_methods_cat]
file_imp_methods_cat.close()
file_imp_methods_num.close()

def validate_order(df_fs, target, ml_method, perc_list, seeds):
    """
    This function prints the results of all the possible orderings of application
    of the imputation methods suggested by the specialized classifiers on the provided
    dataset with the provided percentages of missing values. The results are averaged over
    multiple runs.
    :param df_fs: clean dataset on which the order of suggestions are validated
    Expects four feature columns and one target column
    :param target: name of the class
    :param ml_method: selected downstream ml method
    :param perc_list: percentages of missing values injected in the four columns
    :param seeds: how many experiments to compute on the dataset
    """
    features = list(df_fs.columns)
    features.remove(target)

    """
    plt.figure()
    sns.heatmap(df_fs[features].corr(), annot = True, cmap = 'vlag_r', vmin = -1, vmax = 1)
    #print(df_fs[features].corr())
    plt.show()
    """

    # load the classifiers for numerical and categorical columns
    num_clf = load(f"Classifier/classifiers/classifier_{ml_method}_num.joblib")
    num_scaler = load(f"Classifier/classifiers/scaler_{ml_method}_num.joblib")
    num_mask = load(f"Classifier/classifiers/features_{ml_method}_num.joblib")
    num_mask = np.array(num_mask, dtype=np.bool_)

    cat_clf = load(f"Classifier/classifiers/classifier_{ml_method}_cat.joblib")
    cat_scaler = load(f"Classifier/classifiers/scaler_{ml_method}_cat.joblib")
    cat_mask = load(f"Classifier/classifiers/features_{ml_method}_cat.joblib")
    cat_mask = np.array(cat_mask, dtype=np.bool_)

    scoreboard = np.zeros(24)
    avg_f1_score = np.zeros(24)
    permutations = list(itertools.permutations(range(4), 4))

    # start the repeated experiments
    for seed in range(seeds):
        print(seed)
        np.random.seed(seed)
        df_dirty = df_fs[features].copy()

        # inject the missing values according to the provided list
        for i, perc in enumerate(perc_list):
            p = [perc, 1 - perc]
            col = features[i]
            rand = np.random.choice([True, False], size=df_dirty.shape[0], p=p)
            df_dirty.loc[rand, col] = np.nan

        # compute the suggested imputation methods for each column
        suggested_methods = []
        for col in features:
            column_type = df[col].dtype
            if column_type in ["int64", "float64"]:
                profile = get_features_num(df_dirty, col, num_mask)[None, :]
                profile = num_scaler.transform(profile)
                pred = num_clf.predict(profile)[0]

            else:
                profile = get_features_cat(df_dirty, col, cat_mask)
                profile = np.array([p if not np.isnan(p) else 0 for p in profile])[
                          None, :]
                profile = cat_scaler.transform(profile)
                pred = cat_clf.predict(profile)[0]
            suggested_methods.append(pred)

        print(suggested_methods)

        order_performance_list = []

        # save the performance of each ordering of application of the suggested
        # imputation methods
        for ordering in permutations:
            df_suggestions_order = df_dirty.copy()

            # impute the dataset using the suggestions
            df_suggestions_order = impute_missing_column(df_suggestions_order,"impute_standard", None)
            for i in ordering:
                df_suggestions_order[features[i]] = df_dirty[features[i]].copy()
                values = impute_missing_column(df_suggestions_order.copy(), suggested_methods[i], features[i])[features[i]]
                df_suggestions_order[features[i]] = values.copy()

            df_suggestions_order[target] = df_fs[target]
            # compute the classification metric
            f1_sugg_order = try_classification(df_suggestions_order, target, ml_method, False)
            order_performance_list.append(f1_sugg_order)

        # aggregate the results of the experiments
        perm_ordered = np.array(permutations)[np.argsort(order_performance_list)[::-1]]
        order_performance_list = np.sort(order_performance_list)[::-1]
        for score, perm in enumerate(perm_ordered):
            idx = find_list_index(permutations, perm)
            avg_f1_score[idx] += order_performance_list[score]
            scoreboard[idx] += score

    print("Results")
    avg_f1_score = avg_f1_score/50
    classifica = np.argsort(avg_f1_score)[::-1]
    for position, idx in enumerate(classifica):
        print(f"Pos {position}: {permutations[idx]} with score {scoreboard[idx]} and perf {avg_f1_score[idx]}")


def find_list_index(list_of_lists, target_list):
    """
    utility function for finding the index of a list inside a list of lists
    (used here for finding the index of a permutation among all possible ones)
    :param list_of_lists:
    :param target_list:
    :return: the index of the list
    """
    for idx, sublist in enumerate(list_of_lists):
        flag = True
        for i in range(4):
            if sublist[i] != target_list[i]:
                flag = False
        if flag:
            return idx
    return -1

if __name__ == "__main__":
    name_list = ["wine", "visualizing_galaxy", "student", "consumer"]
    target_list = ["Wine", "binaryClass", "GradeClass", "PurchaseIntent"]
    ml_methods = ["DecisionTree", "LogisticRegression", "KNN", "RandomForest", "AdaBoost", "MLP", "TabNet"]
    for i, name in enumerate(name_list):
        print(name)
        target = target_list[i]
        for ml_method in ml_methods:
            print(ml_method)
            df = get_dataset("Datasets/CSV/", f"{name}.csv")
            if name == "student" or name == "consumer":
                k = 2 # mixed datasets
            else:
                k = 4 # numerical datasets, wine and visualizing_galaxy
            df_fs = fixed_fs_univariate(df, target, cols_to_select=k)
            print(df_fs.columns)
            perc_list = [0.2,0.2,0.2,0.2] # change this for different missing percentages
            validate_order(df_fs, target, ml_method, perc_list, 50)