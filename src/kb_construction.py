import numpy as np
from Feature_selection.feature_selection import feature_selection_univariate, fixed_fs_univariate, remove_corr
from Column_profile_extraction.numerical import get_features_num
from Datasets.get_dataset import get_dataset
from Column_profile_extraction.categorical import get_features_cat
from Imputation.imputation_techniques import impute_missing_column
from Classification.algorithms_class import classification
from itertools import repeat
from multiprocessing import Pool
from utils import dirty_single_column, encoding_categorical_variables
import pandas as pd
import warnings
import time
import traceback
import json
import torch
warnings.filterwarnings("ignore")
# set PyTensor flags cxx to an empty string.
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

# opening files with names of datasets, ml algorithms and imputation methods
file_datasets = open("Datasets/dataset_names.txt", "r")
file_ml_methods = open("Classification/classification_methods.txt", "r")

## =========== NEW EXPERIMENTS WITH NEW IMPUTATION METHODS ============== ##
#file_imp_methods_num = open("Imputation/methods_numerical_column.txt", "r")
#file_imp_methods_cat = open("Imputation/methods_categorical_column.txt", "r")
file_imp_methods_num = open("Imputation/new_methods_numerical_column.txt", "r")
file_imp_methods_cat = open("Imputation/new_methods_categorical_column.txt", "r")

datasets = file_datasets.readlines()
 # removing adult dataset for now
ml_methods = file_ml_methods.readlines()
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_cat = file_imp_methods_cat.readlines()

datasets = [line.strip('\n\r') for line in datasets]
ml_methods = [line.strip('\n\r') for line in ml_methods]
imp_methods_num = [line.strip('\n\r') for line in imp_methods_num]
imp_methods_cat = [line.strip('\n\r') for line in imp_methods_cat]

# this dataframe contains the value of the parameters to train the ml algorithms
df_hyper = pd.read_csv("Hyperparameter_tuning/hyperparameters.csv")

tobe_done_ds = ['car', 'cancer', 'default of credit card clients', 'fried', 'frogs', 'house', 'iris', 'letter', 'mushrooms', 'mv', 'nursery', 'phoneme', 'ringnorm', 'soybean', 'stars', 'wall-robot-navigation']
datasets = [ds for ds in datasets if ds in tobe_done_ds] 
# generate seeds for the different parallel jobs
def generate_seed(n_seed, n_elements):
    seed = []
    seeds = []
    for r in range(0, n_seed):
        for i in range(0, n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds


def _procedure_for_pool(args):
    """
    Helper to execute one parallel job and keep track of its original index.
    """

    torch.set_num_threads(1)
    idx, df, dataset, class_name, column, single_seed = args
    return idx, procedure(df, dataset, class_name, column, single_seed)


# execute the experiments in parallel
def  parallel_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds):
    n_instances_x_job = int(n_instances_tot / n_parallel_jobs)
    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    # write the seeds in the seeds file
    flat_seeds = [x[0] for x in seed]
    new_line_seeds = dataset + "," + column + ","
    for s in flat_seeds:
        new_line_seeds += str(s) + ","
    new_line_seeds = new_line_seeds[:-1] + "\n"
    file_seeds.write(new_line_seeds)

    tasks = [
        (idx, df, dataset, class_name, column, single_seed)
        for idx, single_seed in enumerate(seed)
    ]
    results = [None] * len(tasks)
    total_jobs = len(tasks)
    progress_bar_width = 30

    print(f"Starting parallel jobs for dataset='{dataset}', column='{column}'")
    print(f"Jobs progress [{'-' * progress_bar_width}] 0/{total_jobs}")

    # starts the parallel experiments on the column
    with Pool(processes=n_parallel_jobs) as pool:
        for completed_jobs, (idx, job_result) in enumerate(
            pool.imap_unordered(_procedure_for_pool, tasks),
            start=1
        ):
            results[idx] = job_result
            filled = int(progress_bar_width * completed_jobs / total_jobs)
            bar = "#" * filled + "-" * (progress_bar_width - filled)
            print(f"Jobs progress [{bar}] {completed_jobs}/{total_jobs}", flush=True)

    return results


def sequential_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds):
    """
    Execute the same jobs as parallel_exec, but sequentially.
    The function keeps the same interface and output structure.
    """
    n_instances_x_job = int(n_instances_tot / n_parallel_jobs)
    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    # write the seeds in the seeds file (same format as parallel_exec)
    flat_seeds = [x[0] for x in seed]
    new_line_seeds = dataset + "," + column + ","
    for s in flat_seeds:
        new_line_seeds += str(s) + ","
    new_line_seeds = new_line_seeds[:-1] + "\n"
    file_seeds.write(new_line_seeds)

    results = []
    total_jobs = len(seed)
    progress_bar_width = 30

    print(f"Starting sequential jobs for dataset='{dataset}', column='{column}'")
    print(f"Jobs progress [{'-' * progress_bar_width}] 0/{total_jobs}")

    for completed_jobs, single_seed in enumerate(seed, start=1):
        results.append(procedure(df, dataset, class_name, column, single_seed))
        filled = int(progress_bar_width * completed_jobs / total_jobs)
        bar = "#" * filled + "-" * (progress_bar_width - filled)
        print(f"Jobs progress [{bar}] {completed_jobs}/{total_jobs}", flush=True)

    return results


def load_processed_pairs(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return {}

    try:
        with open(checkpoint_path, "r") as checkpoint_file:
            checkpoint_data = json.load(checkpoint_file)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Unable to read checkpoint file '{checkpoint_path}': {e}")
        return {}

    pairs = checkpoint_data.get("processed_pairs", {})
    processed_pairs = {}

    # New format:
    # {"processed_pairs": {"dataset_a": ["col_1", "col_2"], ...}}
    if isinstance(pairs, dict):
        for dataset_name, columns in pairs.items():
            if isinstance(columns, list):
                processed_pairs[dataset_name] = set(columns)
        return processed_pairs

    # Backward compatibility with old format:
    # {"processed_pairs": ["dataset:column", ...]}
    if isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, str) and ":" in pair:
                dataset_name, column_name = pair.split(":", 1)
                processed_pairs.setdefault(dataset_name, set()).add(column_name)

    return processed_pairs


def save_processed_pairs(checkpoint_path, processed_pairs):
    checkpoint_data = {
        "processed_pairs": {
            dataset_name: sorted(columns)
            for dataset_name, columns in sorted(processed_pairs.items())
        }
    }
    with open(checkpoint_path, "w") as checkpoint_file:
        json.dump(checkpoint_data, checkpoint_file, indent=2)


# procedure for the experiments on a specific column
def procedure(df, dataset, class_name, column, seed):
    features = list(df.columns)
    features.remove(class_name)

    # inject missing values in the df, with different percentages. This data frame contains different versions of the column with missing values (different percentages)
    df_list_no_class = dirty_single_column(df[features], column, class_name, seed)

    # Initialize the results dictionary
    results_experiment = dict()

    column_profile = ()
     
    for i, df_missing in enumerate(df_list_no_class):
        column_type = df[column].dtype

        imputed_datasets = []
        print("Starting imputation on first dirty dataset ", i)
        if column_type in ["int64", "float64", "int32"]:

            # Profile extraction for numerical column with missing values
            column_profile = get_features_num(df_missing, column)

            # impute the numerical column with all the imputation methods
            for imp_method in imp_methods_num:
                # print("[", imp_method, "]")
                current_df = df_missing.copy()
                column_type = current_df[column].dtype
                imputed_df = impute_missing_column(current_df, imp_method,
                                                column)
                imputed_df = encoding_categorical_variables(imputed_df)
                # add the class column back to the imputed dataframe
                imputed_df[class_name] = df[class_name]
                imputed_datasets.append(imputed_df)
                print("Imputation with method ", imp_method, " completed.")

        if column_type in ["bool", "object"]:
            column_profile = get_features_cat(df_missing, column)
            # impute the categorical column with all the imputation methods
            for imp_method in imp_methods_cat:
                print("[", imp_method, "]")   
                current_df = df_missing.copy()
                
                imputed_df = impute_missing_column(current_df, imp_method,
                                                column)
                # print("Imputed dataset shape: ", imputed_df.shape)
                # print("Inputed column unique values: ", imputed_df[column].unique())
                imputed_df = encoding_categorical_variables(imputed_df)

                # add the class column back to the imputed dataframe
                imputed_df[class_name] = df[class_name]
                imputed_datasets.append(imputed_df)
                print("Imputation with method ", imp_method, " completed.")
                # print("Imputed dataset shape: ", imputed_df.shape)
                # print("Inputed dataset columns: ", imputed_df.columns)
                # print("")
        # for imputed in imputed_datasets:
        #     print("Type of imputed dataset: ", type(imputed))
        
        ml_results = dict()
        
        print("Starting ML evaluation...")
        for ml_method in ml_methods:
            print("starting ", ml_method)
            scores = []
            for imputed_df in imputed_datasets:
                # print("Imputed dataset shape: ", imputed_df.shape)
                new_features = list(imputed_df.columns)
                new_features.remove(class_name)
                param = df_hyper[
                    np.logical_and(df_hyper["ml_method"] == ml_method,
                                df_hyper["dataset"] == dataset)][
                    "best_parameter"].values[0]
                ml_score = classification(imputed_df[new_features],
                                        imputed_df[class_name], ml_method,
                                        param)
                scores.append(ml_score)
            ml_results[ml_method] = scores

        results_experiment[i] = [column_profile, ml_results]

        print("/=======================================================/")
        print("Experiment for iteration ", i, " completed.")
        print("/=======================================================/")
    return results_experiment


def write_file(dataset, column, experiment, file):
    print("Writing results for dataset ", dataset, " column ", column)
    for missing_perc in range(10): # there are ten missing percentages
        results_missing_perc = experiment[missing_perc]
        column_profile = results_missing_perc[0]
        ml_results = results_missing_perc[1]

        for ml_index, ml_method in enumerate(ml_methods):
            new_line = dataset + "," + column + ","
            for val in column_profile:
                new_line += str(val) + ","
            new_line += ml_method + ","
            for score in ml_results[ml_method]:
                new_line += str(score) + ","
            new_line = new_line[:-1]
            new_line += "\n"
            file.write(new_line)

def main(reduced_df=False):
    print("Starting knowledge base construction...")
    # print imputation and ml methods used
    print("Imputation methods for numerical columns: ", imp_methods_num)
    print("Imputation methods for categorical columns: ", imp_methods_cat)
    print("ML methods: ", ml_methods)

    path_datasets = "Datasets/CSV/"
    new_exp_path = "Full_ImpExp_ML/"
    checkpoint_path = f"{new_exp_path}processed_pairs_checkpoint.json"
    # sempre multipli
    n_instances_tot = 8
    n_parallel_jobs = 8
    processed_pairs = load_processed_pairs(checkpoint_path)
    n_processed_pairs = sum(len(columns) for columns in processed_pairs.values())
    print(f"Loaded {n_processed_pairs} processed dataset:column pairs from checkpoint.")
    print("Experiment files opened in append mode; headers are written only for empty/new files.")

    # Opening file to save the results (in the new experiments folder)
    files_numerical = []
    files_categorical = []
    for i in range(n_parallel_jobs):
        num_file_path = f"{new_exp_path}experiment_{i+1}_numerical.csv"
        should_write_num_header = (
            not os.path.exists(num_file_path) or os.path.getsize(num_file_path) == 0
        )
        file_num = open(num_file_path, "a")
        num_header_prefix = (
            "name,column_name,n_tuples,missing_perc,uniqueness,"
            "min,max,mean,median,std,skewness,kurtosis,mad,"
            "iqr,p_min,p_max,k_min,k_max,s_min,s_max,entropy,"
            "density,ml_algorithm"
        )
        num_header = num_header_prefix + "," + ",".join(imp_methods_num) + "\n"
        if should_write_num_header:
            file_num.write(num_header)
            print("Numerical file header written.")
        else:
            print(f"Appending to existing numerical file: {num_file_path}")
        files_numerical.append(file_num)

        cat_file_path = f"{new_exp_path}experiment_{i+1}_categorical.csv"
        should_write_cat_header = (
            not os.path.exists(cat_file_path) or os.path.getsize(cat_file_path) == 0
        )
        file_cat = open(cat_file_path, "a")
        cat_header_prefix = (
            "name,column_name,n_tuples,missing_perc,constancy,imbalance,"
            "uniqueness,unalikeability,entropy,density,mean_char,std_char,skewness_char,"
            "kurtosis_char,min_char,max_char,ml_method"
        )
        cat_header = cat_header_prefix + "," + ",".join(imp_methods_cat) + "\n"
        if should_write_cat_header:
            file_cat.write(cat_header)
            print("Categorical file header written.")
        else:
            print(f"Appending to existing categorical file: {cat_file_path}")
        files_categorical.append(file_cat)

    # # Test write on categorial file
    # print("Test write on categorical file.")
    # print("File name: ", files_categorical[0].name)
    # test_line = "test_dataset,test_column,1000,0.1,0.5,0.3,0.2,0.4,1.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,ml_test,impute_standard_test,impute_mode_test,impute_random_test,impute_knn_test,impute_mice_test,impute_logistic_regression_test,impute_random_forest_test,impute_kproto_test\n"
    # files_categorical[0].write(test_line)
    # print("Test write completed.")

    # this files saves the seeds used for each column in the experiments, for reproducibility
    file_seeds = open(f"{new_exp_path}seeds.csv", "w")
    line = "name,column_name,"
    for i in range(n_parallel_jobs):
        line += f"seed_{i},"
    line = line[:-1]
    line += "\n"
    file_seeds.write(line)

    # here starts the main loop on datasets and columns
    print("Datasets to analyze: ", datasets)
    errors = []
    for dataset in datasets:  # removing adult dataset for now
        print("------------" + dataset + "------------")
        df = get_dataset(path_datasets,dataset + ".csv")
        class_name = df.columns[-1]

        #Convert 'str' dtype to 'object' for categorical columns
        str_columns = df.select_dtypes(include=['str']).columns
        df[str_columns] = df[str_columns].astype(object)

        # feature selection
        # df_fs, _, _, _, _ = feature_selection_univariate(df, class_name, perc_num=50, perc_cat=60)
        try:
            df_corr_removed = remove_corr(df, class_name, threshold=0.8)
            df_fs = fixed_fs_univariate(df_corr_removed, class_name)
        except Exception as e:
            print(f"Error during feature selection for dataset {dataset}: {e}")
            errors.append((dataset, "feature_selection", str(e)))
            continue
        columns = list(df_fs.columns)
        columns.remove(class_name)
        print("Columns selected after removing correlated features: ", columns)
        for column in columns:
            if column in processed_pairs.get(dataset, set()):
                print(f"Skipping already processed pair: {dataset}:{column}")
                continue

            try:
                print("ANALYZING ", column)
                if not reduced_df:
                    # print("Using full dataset for experiments.")
                    experiments = parallel_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
                    # experiments = sequential_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
                else:
                    # print("Using reduced dataset for experiments.")
                    experiments = parallel_exec(df_fs, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
                    # experiments = sequential_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
                    # print("Experiments on column ", column, " completed.")
                    # print("Experiments results: ", experiments)

                # write the results of the different experiments in the corresponding files
                column_write_ok = True
                for i, experiment in enumerate(experiments):
                    if df[column].dtype in ["int64","float64"]:
                        print("Writing results on numerical file: ", files_numerical[i].name)
                        try:
                            write_file(dataset, column, experiment, files_numerical[i])
                            print("Write completed.")
                        except Exception as e:
                            print(f"ERROR writing to numerical file {files_numerical[i].name}: {e}")
                            column_write_ok = False
                    else:
                        print("Writing results on categorical file: ", files_categorical[i].name)
                        try:
                            write_file(dataset, column, experiment, files_categorical[i])
                            print("Write completed.")
                        except Exception as e:
                            print(f"ERROR writing to categorical file {files_categorical[i].name}: {e}")
                            column_write_ok = False

                if column_write_ok:
                    processed_pairs.setdefault(dataset, set()).add(column)
                    save_processed_pairs(checkpoint_path, processed_pairs)
                    print(f"Checkpoint updated with pair: {dataset}:{column}")
                else:
                    print(f"Pair not checkpointed due to write errors: {dataset}:{column}")
            except Exception as e:
                tb_last = traceback.extract_tb(e.__traceback__)[-1]
                error_location = f"{tb_last.filename}:{tb_last.lineno} ({tb_last.name})"
                print(
                    f"ERROR in main loop for dataset {dataset}, column {column}: {e} "
                    f"[at {error_location}]"
                )
                errors.append((dataset, column, str(e), error_location))
                raise
        
    # print errors if any
    if errors:
        with open(f"{new_exp_path}errors_log.txt", "w") as error_file:
            for err in errors:
                error_file.write(
                    f"Dataset: {err[0]}, Column: {err[1]}, Error: {err[2]}\n"
                )
        print(f"Errors logged in {new_exp_path}errors_log.txt")

    # closing files
    for i in range(len(files_numerical)):
        files_numerical[i].close()
        files_categorical[i].close()

    file_datasets.close()
    file_imp_methods_cat.close()
    file_imp_methods_num.close()
    file_ml_methods.close()
    file_seeds.close()

if __name__ == "__main__":
    main(reduced_df=True)
