import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.inspection import PartialDependenceDisplay
import random

file_imp_methods_num = open("../Imputation/methods_numerical_column.txt", "r")
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_num = np.array([line.strip('\n\r') for line in imp_methods_num])
file_imp_methods_cat = open("../Imputation/methods_categorical_column.txt", "r")
imp_methods_cat = file_imp_methods_cat.readlines()
imp_methods_cat = np.array([line.strip('\n\r') for line in imp_methods_cat])
file_ml_algorithms = open("../Classification/classification_methods.txt", "r")
ml_algorithms = file_ml_algorithms.readlines()
ml_algorithms = np.array([line.strip('\n\r') for line in ml_algorithms])

file_imp_methods_num.close()
file_imp_methods_cat.close()
file_ml_algorithms.close()


def get_final_df(reduced=True, num=True):
    """
    This function returns a dataset that is the average of the raw experiments in
    the "final files" folder. The profiles and imputation methods scores are averaged
    for each unit of knowledge across the different experiments.
    :param reduced: if the experiments were done on the reduced dataset (i.e.,
    the Knowledge Enrichment process was done on the datasets with feature selection
    applied). Must be True
    :param num: whether to use the knowledge for numerical or categorical columns
    :return: the dataset with average results from the experiments
    """
    if not reduced:
        print("Full dataset experiments no longer available!")
        final_df = None
        """
        if num:
            final_df = pd.read_csv("raw_experiments/experiment_1_numerical.csv")
        else:
            final_df = pd.read_csv("raw_experiments/experiment_1_categorical.csv")
        num_columns = list(final_df.columns)
        num_columns.remove("name")
        num_columns.remove("column_name")
        if num:
            num_columns.remove("ml_algorithm")
        else:
            num_columns.remove("ml_method")
        for i in range(1, 8):
            if num:
                df = pd.read_csv(f"raw_experiments/experiment_{i + 1}_numerical.csv")
            else:
                df = pd.read_csv(f"raw_experiments/experiment_{i + 1}_categorical.csv")
            final_df[num_columns] += df[num_columns]
        final_df[num_columns] /= 8
        """
    else:
        if num:
            final_df = pd.read_csv("all_methods_exp/experiment_1_numerical.csv")
        else:
            final_df = pd.read_csv("all_methods_exp/experiment_1_categorical.csv")
        num_columns = list(final_df.columns)
        num_columns.remove("name")
        num_columns.remove("column_name")
        if num:
            num_columns.remove("ml_algorithm")
        else:
            num_columns.remove("ml_method")
        for i in range(1, 4):
            if num:
                df = pd.read_csv(f"all_methods_exp/experiment_{i + 1}_numerical.csv")
            else:
                df = pd.read_csv(f"all_methods_exp/experiment_{i + 1}_categorical.csv")
            final_df[num_columns] += df[num_columns]
        final_df[num_columns] /= 4
    return final_df.copy()

def tune_binary_classifier(max_iter=100, is_num=True):
    """
    Tries random initializations of a Random Forest classifier and returns the best
    cross-validated set of hyperparameters for all ml methods.
    :param max_iter: maximum number of to-be-tried hyperparameter initializations
    :param is_num: wether fit on numerical units of knowledge or categorical ones
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=5, stop=300, num=40)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    ml_algorithms = ["DecisionTree", "LogisticRegression", "KNN", "RandomForest", "AdaBoost",
                     "MLP", "TabNet"]
    max = np.zeros(len(ml_algorithms))-1 # list that will contain the maximum achieved f1 score
    best_params = {key: [] for key in range(len(ml_algorithms))}
    for iter in range(max_iter):
        print(iter+1)
        params = []
        for key in random_grid:
            params.append(random.choice(random_grid[key]))
        clf = RandomForestClassifier(n_estimators=params[0],
                                     max_features=params[1],
                                     max_depth=params[2],
                                     min_samples_split=params[3],
                                     min_samples_leaf=params[4],
                                     bootstrap=params[5],
                                     random_state=0)

        print(params)
        f1_scores = binary_classifier(clf, is_num)
        print(f1_scores)
        mask = max < f1_scores
        max[mask] = f1_scores[mask]
        for idx in range(len(mask)):
            if mask[idx] == 1:
                print("I changed params for ", idx)
                best_params[idx] = params.copy()

    # print the obtained hyperparameters for each ml method
    for i in range(max.shape[0]):
        print(f"ALGORITHM {i}: ")
        print(max[i])
        print(best_params[i])

def binary_classifier(clf=RandomForestClassifier(random_state=0), is_num=True):
    """
    Trains a set of binary classifier, one for each ml method. The training dataset
    is constructed from the experiments files of the Knowledge Base Enrichment process.
    For each unit of knowledge, the corresponding label is 0 if the number of equivalent
    imputation methods is less than 4, otherwise the label is 1. The classifier learns
    to predict the label given a column profile.
    :param clf: binary classifier to be fit on the data
    :param is_num: whether to use numerical or categorical columns
    :return: the list of f1 scores of the classifier fit for each ml method
    """
    final_df = get_final_df(True, is_num)

    # custom threshold for the computation of the equivalent imputation methods
    if is_num:
        perc = 0.002
    else:
        perc = 0.005
    scores_ml_list = []

    # we fit the classifier for each ml method
    ml_algorithms = ["DecisionTree", "LogisticRegression", "KNN", "RandomForest", "AdaBoost",
                     "MLP", "TabNet"]
    for n, ml_algorithm in enumerate(ml_algorithms):
        print(f"#### {ml_algorithm} ####")

        # dataset preparation
        if is_num:
            df_ml_algorithm = final_df[final_df["ml_algorithm"] == ml_algorithm]
            scores = df_ml_algorithm[imp_methods_num].values
        else:
            df_ml_algorithm = final_df[final_df["ml_method"] == ml_algorithm]
            scores = df_ml_algorithm[imp_methods_cat].values

        sorted_scores_indices = np.argsort(scores, axis=1)[:, ::-1]
        sorted_scores = np.take_along_axis(scores, sorted_scores_indices,axis=1)
        best_methods = []
        for j in range(df_ml_algorithm.shape[0]):
            scores_row = sorted_scores[j, :]
            max_row = np.max(scores_row)
            diff_row = max_row - scores_row
            best_count = np.sum(diff_row <= perc)
            if is_num:
                best_methods.append(list(imp_methods_num[sorted_scores_indices[j, :]][:best_count]))
            else:
                best_methods.append(list(imp_methods_cat[sorted_scores_indices[j, :]][:best_count]))

        labels = []
        for elem in best_methods:
            if len(elem) <= 4:
                labels.append(0)
            else:
                labels.append(1)

        labels = np.array(labels)

        features = list(df_ml_algorithm.columns)
        features.remove("name")
        features.remove("column_name")
        if is_num:
            features.remove("ml_algorithm")
            features = features[:-len(imp_methods_num)]
        else:
            features.remove("ml_method")
            features = features[:-len(imp_methods_cat)]
        df_train = df_ml_algorithm[features]
        df_train = df_train.fillna(0)
        threshold = 0.8
        # remove correlated features, otherwise the pdp plots are misleading

        corr = df_train.corr(numeric_only=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if
                   any(upper[column] > threshold)]

        df_train.drop(df_train[to_drop], axis=1, inplace=True)
        column_names = df_ml_algorithm["column_name"].unique()
        f1_scores = []
        baseline_scores = []

        # here starts the cross-validation
        for column in column_names:
            X_train = df_train[df_ml_algorithm["column_name"]!=column]
            X_test = df_train[df_ml_algorithm["column_name"]==column]
            y_train = labels[df_ml_algorithm["column_name"]!=column]
            y_test = labels[df_ml_algorithm["column_name"]==column]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # baseline computation
            if np.sum(y_train==0) > np.sum(y_train==1):
                y_pred_base = np.zeros(y_test.shape[0])
            else:
                y_pred_base = np.ones(y_test.shape[0])

            f1_sc = f1_score(y_test, y_pred, average="weighted") # model f1-score
            f1_sc_base = f1_score(y_test, y_pred_base, average="weighted") # baseline f1-score
            f1_scores.append(f1_sc)
            baseline_scores.append(f1_sc_base)
        # aggregate the results
        mean_f1_score = np.mean(f1_scores)
        mean_f1_score_base = np.mean(baseline_scores)
        print(f"Model f1: {mean_f1_score}")
        print(f"Baseline f1: {mean_f1_score_base}")
        scores_ml_list.append(mean_f1_score)
    return np.array(scores_ml_list)

def pdp(is_num=True):
    """
    This function constructs the partial dependence plots for each trained binary
    classifier. The binary classifier hyperparameters are those retrieved after
    hyperparameter tuning.
    :param is_num: wheter to use categorical or numerical columns' data
    """
    final_df = get_final_df(True, is_num)
    ml_models = ["DecisionTree", "LogisticRegression", "KNN", "RandomForest", "AdaBoost",
                 "MLP", "TabNet"]

    # Compute a common feature list from the full dataset once to ensure all models
    # use the same features and ax size is consistent
    _all_features = list(final_df.columns)
    _all_features.remove("name")
    _all_features.remove("column_name")
    if is_num:
        _all_features.remove("ml_algorithm")
        _all_features = _all_features[:-len(imp_methods_num)]
    else:
        _all_features.remove("ml_method")
        _all_features = _all_features[:-len(imp_methods_cat)]
    _corr = final_df[_all_features].fillna(0).corr(numeric_only=True)
    _upper = _corr.where(np.triu(np.ones(_corr.shape), k=1).astype(bool))
    _to_drop = set(c for c in _upper.columns if any(_upper[c] > 0.8))
    common_features = [f for f in _all_features if f not in _to_drop]

    n_features = len(common_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 30))
    ax = ax.flatten()[:n_features]
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
    for model in range(7):
        print(model)
        # retrieve the binary classifier
        if is_num:
            if model==0:
                params = [35, 'sqrt', 90, 2, 2, False]
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "DecisionTree"]
            elif model==1:
                params = [269, 'sqrt', 20, 5, 4, True]
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "LogisticRegression"]
            elif model==2:
                params = [20, 'log2', 110, 5, 1, True]
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "KNN"]
            elif model==3:
                params = [103, 'log2', 30, 5, 1, True]
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "RandomForest"]
            elif model==4:
                params = [12, 'sqrt', 50, 2, 4, False]
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "AdaBoost"]
            elif model==5:
                params = [12, 'sqrt', 50, 2, 4, False]  # TODO: fill after tuning
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "MLP"]
            else:
                params = [50, 'log2', 10, 10, 2, True]  # TODO: fill after tuning
                df_ml_algorithm = final_df[final_df["ml_algorithm"] == "TabNet"]

        else:
            if model == 0:
                params = [5, 'log2', 10, 2, 4, True] # [12, 'log2', 30, 5, 2, False]
                df_ml_algorithm = final_df[final_df["ml_method"] == "DecisionTree"]
            elif model == 1:
                params = [12, 'sqrt', 90, 5, 2, True] # [12, 'log2', 110, 5, 4, False]
                df_ml_algorithm = final_df[final_df["ml_method"] == "LogisticRegression"]
            elif model == 2:
                params = [292, 'log2', 40, 10, 1, True]
                df_ml_algorithm = final_df[final_df["ml_method"] == "KNN"]
            elif model == 3:
                params = [292, 'log2', 10, 2, 1, False]
                df_ml_algorithm = final_df[final_df["ml_method"] == "RandomForest"]
            elif model == 4:
                params = [300, 'sqrt', None, 2, 1, False]
                df_ml_algorithm = final_df[final_df["ml_method"] == "AdaBoost"]
            elif model == 5:
                params = [141, 'sqrt', 60, 2, 2, False]  # TODO: fill after tuning
                df_ml_algorithm = final_df[final_df["ml_method"] == "MLP"]
            else:
                params = [42, 'sqrt', 50, 2, 2, False]  # TODO: fill after tuning
                df_ml_algorithm = final_df[final_df["ml_method"] == "TabNet"]

        clf = RandomForestClassifier(n_estimators=params[0],
                                     max_features=params[1],
                                     max_depth=params[2],
                                     min_samples_split=params[3],
                                     min_samples_leaf=params[4],
                                     bootstrap=params[5],
                                     random_state=0
                                     )

        # prepare the dataset
        if is_num:
            scores = df_ml_algorithm[imp_methods_num].values
        else:
            scores = df_ml_algorithm[imp_methods_cat].values
        sorted_scores_indices = np.argsort(scores, axis=1)[:, ::-1]
        sorted_scores = np.take_along_axis(scores, sorted_scores_indices, axis=1)
        best_methods = []
        for j in range(df_ml_algorithm.shape[0]):
            scores_row = sorted_scores[j, :]
            max_row = np.max(scores_row)
            diff_row = max_row - scores_row
            best_count = np.sum(diff_row <= 0.002)
            if is_num:
                best_methods.append(list(imp_methods_num[sorted_scores_indices[j, :]][:best_count]))
            else:
                best_methods.append(list(imp_methods_cat[sorted_scores_indices[j, :]][:best_count]))

        labels = []
        for elem in best_methods:
            if len(elem) <= 4:
                labels.append(0)
            else:
                labels.append(1)

        labels = np.array(labels)

        df_train = df_ml_algorithm[common_features].fillna(0)

        # train the model
        clf.fit(df_train, labels)

        # print the pdp plot
        if is_num:
            plot_num(clf, ml_models, model, df_train, ax, colors)
        else:
            plot_cat(clf, ml_models, model, df_train, ax, colors)

    plt.savefig(f"../../results and figures/PDP/pdp_{is_num}.png")

def plot_num(clf, ml_models, model, df_train, ax, colors):
    d = PartialDependenceDisplay.from_estimator(clf, df_train, range(len(list(df_train.columns))), ax=ax, random_state=0)
    d.plot(ax=ax, line_kw={"label":ml_models[model], "color": colors[model]})
    for i, a in enumerate(d.axes_.flatten()):
        if a is not None:
            a.grid()
            if i in [0,4,5,7,8]:
                a.set_xscale("log")
            if i == 3:
                a.set_xlim([-100,26])
            a.set_ylim([0.2,0.8])

def plot_cat(clf, ml_models, model, df_train, ax, colors):
    d = PartialDependenceDisplay.from_estimator(clf, df_train, range(len(list(df_train.columns))), ax=ax, random_state=0)
    d.plot(ax=ax, line_kw={"label": ml_models[model], "color": colors[model]})
    for i, a in enumerate(d.axes_.flatten()):
        if a is not None:
            a.grid()
            if i == 0:
                a.set_xscale("log")
            a.set_ylim([0.2, 0.8])

if __name__ == "__main__":
    # tuning the binary classifiers
    #tune_binary_classifier(is_num=False)
    #tune_binary_classifier(is_num=True)

    # for baseline computation
    binary_classifier(clf=RandomForestClassifier(), is_num=False)
    binary_classifier(clf=RandomForestClassifier(), is_num=True)

    # show partial dependency plots
    plt.rcParams['axes.labelsize'] = 22
    #pdp(is_num=True)
    #pdp(is_num=False)
