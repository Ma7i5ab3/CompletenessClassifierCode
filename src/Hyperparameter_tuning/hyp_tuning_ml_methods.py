from Classification.algorithms_class import classification
from Datasets.get_dataset import get_dataset
from utils import encoding_categorical_variables
import numpy as np
from Feature_selection.feature_selection import remove_corr, fixed_fs_univariate
import time
import warnings
warnings.filterwarnings("ignore")

def search_parameters_classification():
    """
    This function saves on a file the best hyperparameter for each
    ml method on the clen datasets. This hyperparameter will be retrieved in
    the Knowledge Base enrichment process to train the ml method on the imputed
    dataset. This saves time and computation, otherwise we should do tuning for
    each imputed version of the dataset in the KB enrichment, which is not feasible.
    """
    path_datasets = "../Datasets/CSV/"
    file_datasets = open("../Datasets/dataset_names.txt", "r")
    file_classif_methods = open("../Classification/classification_methods.txt", "r")
    file_hyperparams = open("hyperparameters.csv", "w")
    file_hyperparams.write("ml_method,dataset,best_parameter\n")

    dt_params = [5,10,15,20,25]
    lr_params = [0.001, 0.01, 0.1, 1, 10]
    knn_params = [3,5,7,9,11]
    rf_params = [5,10,15,20,25]
    ada_params = [30,40,50,60,70]
    svc_params = [0.001, 0.01, 0.1, 1, 10]
    # Deep models: tune learning rate, while max epochs is fixed in the trainer
    mlp_params = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    tabnet_params = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    ft_params = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    classif_methods = file_classif_methods.readlines()
    classif_methods = [line.strip('\n\r') for line in classif_methods]
    datasets = file_datasets.readlines()
    datasets = [line.strip('\n\r') for line in datasets]
    for classif_method in classif_methods:
        print("Started method: ", classif_method)
        if classif_method == "DecisionTree":
            hyp_list = dt_params
        elif classif_method == "LogisticRegression":
            hyp_list = lr_params
        elif classif_method == "KNN":
            hyp_list = knn_params
        elif classif_method == "RandomForest":
            hyp_list = rf_params
        elif classif_method == "AdaBoost":
            hyp_list = ada_params
        elif classif_method == "MLP":
            hyp_list = mlp_params
        elif classif_method == "TabNet":
            hyp_list = tabnet_params
        elif classif_method == "FTTransformer":
            hyp_list = ft_params
        else:
            hyp_list = svc_params
        start = time.time()
        for dataset in datasets:
            print(dataset)
            df = get_dataset(path_datasets, dataset + ".csv")
            class_name = df.columns[-1]

            #df_fs, _, _, _, _ = feature_selection_univariate(df, class_name,perc_num=50,perc_cat=60)
            df_corr_removed = remove_corr(df, class_name, threshold=0.8)
            df_fs = fixed_fs_univariate(df_corr_removed, class_name)
            features = list(df_fs.columns)
            features.remove(class_name)
            X = df_fs[features]
            y = df_fs[class_name]
            X = encoding_categorical_variables(X)
            score_list = []
            for param in hyp_list:
                score = classification(X, y, classif_method, param)
                score_list.append(score)

            best_param = hyp_list[np.argmax(score_list)]
            new_line = classif_method+","+dataset+","+str(best_param)+"\n"
            file_hyperparams.write(new_line)
        print(classif_method + " took " + str(time.time()-start) + " seconds")
    file_datasets.close()
    file_hyperparams.close()
    file_classif_methods.close()

if __name__=='__main__':
    search_parameters_classification()
