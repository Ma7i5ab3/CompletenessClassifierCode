import numpy as np
import pandas as pd
import kmodes
from sklearn import linear_model
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skfuzzy import cmeans, cmeans_predict
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from utils import encoding_categorical_variables, restore_nans
from sklearn.neighbors import KNeighborsClassifier

# ===================================================================
# New Imputation methods classes
# ===================================================================
from hyperimpute.plugins.imputers import Imputers
from fancyimpute import SoftImpute
from xgbimputer import XGBImputer
from catboost import CatBoostRegressor, CatBoostClassifier
from autoimpute.imputations import MiceImputer
import miceforest as mf
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class no_impute:
    def __init__(self):
        self.name = 'No imputation'

    def fit(self, df):
        return df

class impute_standard:
    def __init__(self):
        self.name = 'Standard'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Missing")
        return df

class drop:
    def __init__(self):
        self.name = 'Drop'

    def fit_cols(self, df):
        df = df.dropna(axis=1, how='any')
        return df

    def fit_rows(self, df):
        df = df.dropna(axis=0, how='any')
        return df

class impute_mean:
    def __init__(self):
        self.name = 'Mean'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
        return df

    def fit_mode(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_mode:
    def __init__(self):
        self.name = 'Mode'

    def fit(self, df):
        df = df.copy()
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_median():
    def __init__(self):
        self.name = 'Median'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].median())
        return df

    def fit_mode(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        return df

class impute_knn():
    def __init__(self):
        self.name = 'KNN'

    def fit(self, df, missing_column, n_neighbors=5):
        type_missing = df.dtypes[missing_column]
        X = df.copy()
        if type_missing in ["int64", "float64"]:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            X = encoding_categorical_variables(X)
            columns = X.columns
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            df_m = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(X)))
            df_m.columns = columns
            return df_m

        elif type_missing in ["bool","object"]:
            train_columns = list(X.columns)
            train_columns.remove(missing_column)
            target = X[missing_column]

            X = encoding_categorical_variables(X[train_columns])
            train_columns = X.columns
            X[missing_column] = target

            x_train = X.loc[target.notna(),train_columns]
            to_impute = X.loc[target.isna(), train_columns]
            y_train = target[target.notna()]

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            to_impute = scaler.transform(to_impute)

            imputer = KNeighborsClassifier(n_neighbors=n_neighbors)
            imputer.fit(x_train, y_train)
            X.loc[target.isna(),missing_column] = imputer.predict(to_impute)
            return X

class impute_mice:
    def __init__(self):
        self.name = 'Mice_mine'

    def fit(self, df, missing_column, estimator):
        type_missing = df.dtypes[missing_column]
        X = df.copy()
        if type_missing in ["int64", "float64"]:
            # one hot encoding
            X = encoding_categorical_variables(X)
            columns = X.columns.copy()
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            imputer = IterativeImputer(max_iter=100, skip_complete=True, estimator=estimator, random_state=0)
            X = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(X)), columns=columns)
            return X

        elif type_missing in ["bool", "object"]:

            # one hot encoding
            fully_available_columns = list(X.columns)
            fully_available_columns.remove(missing_column)
            target = X[missing_column]
            X = encoding_categorical_variables(X[fully_available_columns])
            columns = list(X.columns)
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=columns)
            X[missing_column] = target
            columns.append(missing_column)
            # encode the missing column only for avoiding runtime errors in the IterativeImputer object
            oe = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=np.nan
            )
            missing_col_values = X[missing_column].to_numpy(copy=True).reshape(-1, 1)
            oe.fit(missing_col_values)
            X[missing_column] = oe.transform(missing_col_values).ravel()

            imputer = IterativeImputer(
                estimator=estimator, max_iter=100,
                initial_strategy="most_frequent", skip_complete=True, random_state=0)
            X = pd.DataFrame(imputer.fit_transform(X), columns=columns)
            columns.remove(missing_column)
            X[columns] = scaler.inverse_transform(X[columns])
            encoded_values = X[missing_column].to_numpy(copy=True).reshape(-1, 1)
            X[missing_column] = oe.inverse_transform(encoded_values).ravel()
            return X

class impute_random:
    def __init__(self):
        self.name = 'Random'

    def fit(self, df):
        for col in df.columns:
            number_missing = df[col].isnull().sum()
            observed_values = df.loc[df[col].notnull(), col]
            df.loc[df[col].isnull(), col] = np.random.choice(observed_values, number_missing, replace=True)
        return df

    def fit_single_column(self, df, col):
        df = df.copy()
        number_missing = df[col].isnull().sum()
        observed_values = df.loc[df[col].notnull(), col]
        df.loc[df[col].isnull(), col] = np.random.choice(observed_values, number_missing, replace=True)
        return df


class impute_linear_regression:
    def __init__(self):
        self.name = 'Linear Regression'

    def fit(self, df, missing_column):
        X = df.copy()
        target = X[missing_column].copy()
        fully_available_columns = list(X.columns)
        fully_available_columns.remove(missing_column)
        X = encoding_categorical_variables(X[fully_available_columns])
        X[missing_column] = target

        # here starts the imputation for the single column
        columns_for_imputation = list(X.columns)
        columns_for_imputation.remove(missing_column)

        features = X[columns_for_imputation]
        target = X[missing_column]

        X_train = features[target.notna()]
        y_train = target[target.notna()]
        mean_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train = (y_train - mean_y)/std_y

        to_impute = features[target.isna()]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        to_impute = scaler.transform(to_impute)

        imputer = linear_model.LinearRegression()
        imputer.fit(X_train, y_train)

        X.loc[target.isna(), missing_column] = imputer.predict(to_impute)*std_y + mean_y
        return X


class impute_logistic_regression:
    def __init__(self):
        self.name = 'Logistic Regression'

    def fit(self, df, missing_column, C=1):
        X = df.copy()
        target = X[missing_column].copy()
        fully_available_columns = list(X.columns)
        fully_available_columns.remove(missing_column)
        X = encoding_categorical_variables(X[fully_available_columns])
        X[missing_column] = target

        # here starts the imputation for the single column
        columns_for_imputation = list(X.columns)
        columns_for_imputation.remove(missing_column)

        features = X[columns_for_imputation]
        target = X[missing_column]

        X_train = features[target.notna()]
        y_train = target[target.notna()]

        to_impute = features[target.isna()]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        to_impute = scaler.transform(to_impute)

        imputer = linear_model.LogisticRegression(max_iter=1000, C=C, random_state=0)
        imputer.fit(X_train, y_train)

        X.loc[target.isna(), missing_column] = imputer.predict(to_impute)
        return X


class impute_random_forest:
    def __init__(self):
        self.name = 'Random Forest'

    def fit(self, df, missing_column, max_depth=20):
        X = df.copy()
        target = X[missing_column].copy()
        fully_available_columns = list(X.columns)
        fully_available_columns.remove(missing_column)
        X = encoding_categorical_variables(X[fully_available_columns])
        X[missing_column] = target

        # here starts the imputation for the single column
        columns_for_imputation = list(X.columns)
        columns_for_imputation.remove(missing_column)

        features = X[columns_for_imputation]
        target = X[missing_column]

        X_train = features[target.notna()]
        y_train = target[target.notna()]

        to_impute = features[target.isna()]

        type_missing = X.dtypes[missing_column]
        if type_missing == 'int64' or type_missing == 'float64':
            imputer = RandomForestRegressor(max_depth=max_depth, random_state=42)
        else:
            imputer = RandomForestClassifier(max_depth=max_depth, random_state=42)
        imputer.fit(X_train, y_train)

        X.loc[target.isna(), missing_column] = imputer.predict(to_impute)
        return X


class impute_clustering():
    def __init__(self):
        self.name = 'Clustering'

    def fit_num(self, df, missing_column, n_clusters=5, m=1.5):

        df = df.copy()
        # here starts the imputation for the single column
        features = df.copy()
        target = df[missing_column]
        columns_to_encode = list(features.columns)
        columns_to_encode.remove(missing_column)
        encoded_columns_df = encoding_categorical_variables(features[columns_to_encode])

        X = encoded_columns_df.copy()
        X[missing_column] = target.copy()

        # scale the dataset and cluster the fully available data
        X_train = X[target.notna()]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_train = X_train.T
        centroids,_,_,_,_,_,_ = cmeans(X_train, n_clusters, m=m, error=0.001, maxiter=100000)
        centroid_miss_column_coeff = np.array([centroid[-1] for centroid in centroids])

        # predict cluster attributions for incomplete data
        to_impute = encoded_columns_df.copy()
        to_impute = to_impute[target.isna()]
        # intermediate_value = np.random.choice(centroid_miss_column_coeff,to_impute.shape[0])
        missing_column_mean = np.nanmean(target)
        intermediate_value = [missing_column_mean for i in range(to_impute.shape[0])]
        to_impute[missing_column] = intermediate_value
        to_impute = scaler.transform(to_impute)
        to_impute = to_impute.T
        memberships = cmeans_predict(to_impute, centroids, m=m, error=0.001, maxiter=1000)[0]
        # impute data based on centroids and cluster attributions
        imputed_values = (memberships.T @ centroid_miss_column_coeff)*scaler.scale_[-1] + scaler.mean_[-1]
        df.loc[target.isna(), missing_column] = imputed_values
        return df

    def fit_cat(self, df, missing_column, n_clusters=4):
        # here starts the imputation for the single column
        # X = impute_random().fit_single_column(df.copy(), missing_column)
        X = impute_mode().fit(df.copy())
        cat = list(df.select_dtypes(include=['bool', 'object']).columns)
        num = list(df.select_dtypes(include=['int64', 'float64']).columns)

        missing_column_index = 0
        for i in range(len(cat)):
            if missing_column == cat[i]:
                missing_column_index = i

        cat_indices = [df.columns.get_loc(col) for col in cat]
        for i in range(1):
            if len(num) != 0:
                model = KPrototypes(n_clusters=n_clusters, max_iter=10, init="random")
            else:
                model = KModes(n_clusters=n_clusters, max_iter=10)
            model.fit(X, categorical=cat_indices)
            labels = model.predict(X[df[missing_column].isna()], categorical=cat_indices)
            centroids_values = model.cluster_centroids_[:,len(num)+missing_column_index]
            imputed_values = np.array([centroids_values[label] for label in labels])
            X.loc[df[missing_column].isna(), missing_column] = imputed_values
        return X
    
# ===================================================================
                        # New methods #
# ===================================================================

# Just for numeric features
class impute_expectation_maximization():
    def __init__(self):
        self.name = 'Expectation Maximization'

    def fit(self, df, missing_column):
        X = df.copy()
        if missing_column is None or missing_column not in X.columns:
            return X

        if not pd.api.types.is_numeric_dtype(X[missing_column]):
            return X

        missing_mask = X[missing_column].isna()
        if not missing_mask.any():
            return X

        # If the target column is fully missing there is no signal to estimate from.
        if X[missing_column].notna().sum() == 0:
            X.loc[missing_mask, missing_column] = 0.0
            return X

        # HyperImpute EM expects a 2D numeric table; passing a Series can trigger
        # scalar/0d-array paths with newer NumPy versions.
        numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
        if missing_column not in numeric_columns:
            return X

        # Univariate EM is unstable in the HyperImpute plugin implementation.
        if len(numeric_columns) == 1:
            fill_value = X[missing_column].mean()
            X.loc[missing_mask, missing_column] = fill_value
            return X

        numeric_df = X[numeric_columns].astype(float)
        em_fallback = IterativeImputer(
            estimator=BayesianRidge(),
            random_state=0,
            max_iter=25,
        )

        # HyperImpute EM currently breaks on NumPy 2.x for some inputs.
        numpy_major = int(str(np.__version__).split(".")[0])
        if numpy_major < 2:
            plugin = Imputers().get("EM")
            try:
                imputed_numeric = plugin.fit_transform(numeric_df)
            except Exception:
                imputed_numeric = em_fallback.fit_transform(numeric_df)
        else:
            imputed_numeric = em_fallback.fit_transform(numeric_df)

        if not isinstance(imputed_numeric, pd.DataFrame):
            imputed_numeric = pd.DataFrame(
                imputed_numeric, index=X.index, columns=numeric_columns
            )
        X.loc[missing_mask, missing_column] = imputed_numeric.loc[missing_mask, missing_column]
        X[missing_column] = X[missing_column].fillna(X[missing_column].mean())
        return X

# Only numerical features
class impute_soft_imputer():
    def __init__(self):
        self.name = 'Soft Imputer'

    def fit(self, df):
        # print("Inside soft imputer")
        # X_incomplete_normalized = BiScaler().fit_transform(df)
        original_df = df.copy()
        encoded_df = encoding_categorical_variables(df)
        columns = list(encoded_df.columns)
        index = encoded_df.index

        imputer = SoftImpute(verbose=False)
        try:
            imputed_values = imputer.fit_transform(encoded_df)
        except TypeError as exc:
            # fancyimpute may still call check_array(force_all_finite=...) which is
            # unsupported in newer sklearn versions. Fall back to a stable imputer.
            if "force_all_finite" not in str(exc):
                raise
            fallback_imputer = IterativeImputer(
                estimator=BayesianRidge(),
                random_state=0,
                max_iter=25,
            )
            imputed_values = fallback_imputer.fit_transform(encoded_df)

        imputed_df = pd.DataFrame(imputed_values, columns=columns, index=index)

        # Keep original columns and order if encoding did not expand features.
        if len(imputed_df.columns) == len(original_df.columns):
            imputed_df.columns = original_df.columns

        return imputed_df

# Works with both types of features
class impute_xgb_imputer():
    def __init__(self):
        self.name = 'XGB Imputer'

    def _fallback_single_column_fill(self, df, column_missing):
        df_filled = df.copy()
        if pd.api.types.is_numeric_dtype(df_filled[column_missing]):
            fill_value = df_filled[column_missing].mean()
        else:
            mode_values = df_filled[column_missing].mode(dropna=True)
            fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "__missing__"
        df_filled[column_missing] = df_filled[column_missing].fillna(fill_value)
        return df_filled

    def _remap_categorical_indices(self, columns, column_missing, categorical_features_index):
        missing_idx = columns.get_loc(column_missing)
        predictor_cat_indices = []
        for idx in categorical_features_index:
            if idx < 0 or idx >= len(columns):
                continue
            if idx == missing_idx:
                continue
            predictor_cat_indices.append(idx if idx < missing_idx else idx - 1)
        return sorted(set(predictor_cat_indices))

    def _iterative_xgb_impute(self, df, columns, encoder=None):
        """Run IterativeImputer with XGBRegressor, optionally wrapping with ordinal encoding."""
        if encoder is not None:
            df_work = pd.DataFrame(encoder.fit_transform(df), columns=columns)
        else:
            df_work = df
        imputed = IterativeImputer(
            estimator=XGBRegressor(n_estimators=100, random_state=0),
            random_state=0
        ).fit_transform(df_work)
        result = pd.DataFrame(imputed, columns=columns)
        if encoder is not None:
            result = pd.DataFrame(
                encoder.inverse_transform(result.round().clip(0).astype(int).values.copy()),
                columns=columns
            )
        return result

    def fit(self, df, column_missing, categorical_features_index, replace_values_back=True):
        print(f"Dataset head before XGB Imputer: {df.head()}")
        columns = df.columns
        if column_missing not in columns:
            print(f"Warning: Missing column '{column_missing}' not found in DataFrame. Returning original DataFrame.")
            return df
        if len(columns) <= 1:
            print("Warning: DataFrame has 1 or fewer columns. XGB Imputer cannot be applied. Falling back to simple fill.")
            return self._fallback_single_column_fill(df, column_missing)

        # check if categorical features index is empty use IterativeImputer with XGBRegressor as estimator as equivalent to the XGBImputer for numeric features
        if len(categorical_features_index) == 0:
            print("No categorical features detected. Using IterativeImputer with XGBoost estimator.")
            return self._iterative_xgb_impute(df, columns)

        # XGBImputer crashes when all features are categorical (numerical_features_index is empty).
        # Fall back to IterativeImputer with ordinal encoding in that case.
        all_categorical = set(categorical_features_index) >= set(range(len(columns)))
        if all_categorical:
            print("All features are categorical. Falling back to IterativeImputer with XGBoost estimator.")
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            return self._iterative_xgb_impute(df, columns, encoder=enc)

        imputer = XGBImputer(
            categorical_features_index=categorical_features_index,
            replace_categorical_values_back=replace_values_back
        )

        # If type of column is object or bool, then it is categorical
        try:
            if df.dtypes[column_missing] in ["object", "bool"]:
                print(f"Dataframe head: {df.head()}, with categorical features index: {categorical_features_index}")
                X = imputer.fit_transform(df.to_numpy())
                df = pd.DataFrame(X)
            else:
                df = np.array(df)
                # print("We are inside the class. Input shape: ", df.shape)
                df = pd.DataFrame(imputer.fit_transform(df))
        except Exception as exc:
            print("XGB Imputer failed with exception: ", exc)
            if "Data must has at least 1 column" in str(exc):
                fallback_df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df, columns=columns)
                return self._fallback_single_column_fill(fallback_df, column_missing)
            raise
            
        df.columns = columns

        valid_full_categorical_indices = sorted(
            set(idx for idx in categorical_features_index if 0 <= idx < len(columns))
        )
        numerical_indices = list(set(range(len(columns))) - set(valid_full_categorical_indices))

        for idx in numerical_indices:
            col_name = columns[idx]
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                
        return df

class impute_catboost():
    def __init__(self):
        self.name = 'CatBoost Imputer'

    def fit(self, df, missing_column):

        type_missing = df.dtypes[missing_column]
        # missing_column = df[missing_column]
        # print("Type missing: ", type_missing)
        X = df.copy()   

        # Select categorical features from the dataset
        cat_features = list(df.select_dtypes(include=["object", "bool"]).columns)

        if type_missing in ["int64", "float64"]:
            # Use CatBoostRegressor
            fully_available_samples = X[X[missing_column].notnull()]
            missing = X[X[missing_column].isnull()]

            X_train = fully_available_samples.drop(columns = [missing_column])
            # print("X_train type: ", type(X_train))
            # print("X_train shape: ", X_train.shape)
            y_train = fully_available_samples[missing_column]

            X_pred = missing.drop(columns = [missing_column])

            # Up to here we have the training set in X_train and y_train and the uncomplete samples in X_pred

            imputer = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                loss_function='RMSE',
                verbose=False,
                random_seed=42
            )

            if len(fully_available_samples) > 1 and len(missing) > 0:
                imputer.fit(X_train, y_train, cat_features=cat_features)
                # print(type(df))
                df.loc[df[missing_column].isnull(), missing_column] = imputer.predict(X_pred).flatten()
                df = pd.DataFrame(df)
                return df
            
            df = pd.DataFrame(columns=df.columns)
            return df
            
        elif type_missing in ["bool", "object"]:
            cat_features = [feat for feat in cat_features if feat != missing_column]
            # Use CatBoostClassifier
            fully_available_samples = X[X[missing_column].notnull()]
            missing = X[X[missing_column].isnull()]

            # # encode categorical variables
            # fully_available_samples = encoding_categorical_variables(fully_available_samples)
            # print("Fully available samples after encoding: ", fully_available_samples.head())

            # missing = encoding_categorical_variables(missing)

            X_train = fully_available_samples.drop(columns = [missing_column])
            y_train = fully_available_samples[missing_column]

            X_pred = missing.drop(columns = [missing_column])

            imputer = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                loss_function='MultiClass',
                verbose=False,
                random_seed=42
            )

            if len(fully_available_samples) > 1 and len(missing) > 0:
                imputer.fit(X_train, y_train, cat_features=cat_features)
                df.loc[df[missing_column].isnull(), missing_column] = imputer.predict(X_pred).flatten()
                df = pd.DataFrame(df)
                return df

            # print("Debug")
            return 0

# Both kind of features
class impute_rfi():
    def __init__(self):
        self.name = 'Random Forest Imputer'

    def fit(self, df, missing_column):
        # Get categorical features index
        categorical_features = list(df.select_dtypes(include=["object", "bool"]).columns)
        categorical_features_index = [df.columns.get_loc(col) for col in categorical_features]
        
        for col in df.select_dtypes(include=["object" ,"bool"]).columns:
            df[col] = df[col].astype('category')
        
        kernel = mf.ImputationKernel(
            data=df,
            # save_all_iterations_data=True,
            random_state=42,
            mean_match_candidates=0
        )

        kernel.mice(3)

        df = kernel.complete_data()

        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype('object')
        
        return df

# To test
class impute_autoimpute():
    def __init__(self):
        self.name = 'Autoimpute'

    '''
    def fit(self, df):
        mice = MiceImputer(return_list=True)
        mice = mice.fit(df)
        df = pd.DataFrame(mice.transform(df))
        return df
    
    '''
    def fit(self, df):
        # Use sklearn's IterativeImputer (MICE) instead of autoimpute's MiceImputer,
        # which is incompatible with pandas 3.0 (ambiguous Series truth value in checks.py).
        df_encoded = encoding_categorical_variables(df.copy())
        # convert boolean columns to float
        for col in df_encoded.select_dtypes(include=["bool"]).columns:
            df_encoded[col] = df_encoded[col].astype(float)
        # Execute MICE imputation
        imputer = IterativeImputer(estimator=LinearRegression(), random_state=0)
        imputed_array = imputer.fit_transform(df_encoded)
        return pd.DataFrame(imputed_array, columns=df_encoded.columns)

# Both GAIN and VAE work with both types of features
class impute_gain():
    def __init__(self):
        self.name = 'GAIN Impute'

    def fit(self, df, missing_column):
        plugin = Imputers().get("gain")

        if df[missing_column].dtype in ["int64", "float64"]:
            # encode categorical variables
            X = df.copy()
            target = X[missing_column]
            X = X.drop(columns=[missing_column])
            X = encoding_categorical_variables(X)
            X[missing_column] = target
            X = X.astype(float)

            columns = list(X.columns)

            df = plugin.fit_transform(X)
            df = pd.DataFrame(df)

            # maps back the original columns
            df.columns = columns

            return df
        
        # Deprecated else branch
        # else:
        #     df = encoding_categorical_variables(df)
        #     df = restore_nans(df)
        #     df = df.astype(float)
        #     columns = list(df.columns)
        #     print("Data types after encoding: ", df.head())
        #     df = plugin.fit_transform(df)
        #     df = pd.DataFrame(df)
        #     df.columns = columns

        #     # Take the max of the "probabilities" to decide the final category, given the one hot encoding of the missing categorical variable
        #     col_prefix = missing_column + "_"
        #     targets = df.columns[df.columns.str.startswith(col_prefix)]
        #     print("Targets: ", targets)
        #     for index in df.index:
        #         max_value = -1
        #         selected_col = None
        #         for col in targets:
        #             if df.loc[index, col] > max_value:
        #                 max_value = df.loc[index, col]
        #                 selected_col = col
        #         # Set all target columns to 0
        #         for col in targets:
        #             df.loc[index, col] = 0
        #         # Set the selected column to 1
        #         if selected_col is not None:
        #             df.loc[index, selected_col] = 1
        #     return df
        
class impute_vae():
    def __init__(self):
        self.name = 'VAE Impute'

    def fit(self, df, missing_column):
        plugin = Imputers().get("miwae")

        if df[missing_column].dtype in ["int64", "float64"]:
            # encode categorical variables
            X = df.copy()
            target = X[missing_column]
            X = X.drop(columns=[missing_column])
            X = encoding_categorical_variables(X)
            X[missing_column] = target
            X = X.astype(float)

            columns = list(X.columns)

            df = plugin.fit_transform(X)
            df = pd.DataFrame(df)

            # maps back the original columns
            df.columns = columns

            return df
        
        # Deprecated else branch
        # else:
        #     df = encoding_categorical_variables(df)
        #     df = restore_nans(df)
        #     df = df.astype(float)
        #     columns = list(df.columns)
        #     print("Data types after encoding: ", df.head())
        #     df = plugin.fit_transform(df)
        #     df = pd.DataFrame(df)
        #     df.columns = columns

        #     # Take the max of the "probabilities" to decide the final category, given the one hot encoding of the missing categorical variable
        #     col_prefix = missing_column + "_"
        #     targets = df.columns[df.columns.str.startswith(col_prefix)]
        #     print("Targets: ", targets)
        #     for index in df.index:
        #         max_value = -1
        #         selected_col = None
        #         for col in targets:
        #             if df.loc[index, col] > max_value:
        #                 max_value = df.loc[index, col]
        #                 selected_col = col
        #         # Set all target columns to 0
        #         for col in targets:
        #             df.loc[index, col] = 0
        #         # Set the selected column to 1
        #         if selected_col is not None:
        #             df.loc[index, selected_col] = 1
        #     return df

# Both
class impute_mlp():
    def __init__(self):
        self.name = 'MLP Impute Manual'

    def fit(self, df: pd.DataFrame, missing_column) -> pd.DataFrame:
        X = df.copy()

        # if missing column is numerical
        if df[missing_column].dtype in ["int64", "float64"]:

            X = encoding_categorical_variables(X)

            print("Data after encoding: ", X.head())
    
            mlp_estimator = MLPRegressor(
                random_state=42, 
                solver='adam', 
                max_iter=100, 
                hidden_layer_sizes=(100, 50), # Example architecture
                early_stopping=True
            )

            # Split the data into training and prediction sets
            train_data = X[X[missing_column].notnull()]
            predict_data = X[X[missing_column].isnull()]

            if len(predict_data) == 0:
                return df

            X_train = train_data.drop(columns=[missing_column])
            # X_train = encoding_categorical_variables(X_train)
            y_train = train_data[missing_column]

            X_predict = predict_data.drop(columns=[missing_column])
            # X_predict = encoding_categorical_variables(X_predict)

            print("X_train shape: ", X_train.shape)
            print("y_train shape: ", y_train.shape)

            # Fit the model and predict missing values
            mlp_estimator.fit(X_train, y_train)
            predicted_values = mlp_estimator.predict(X_predict)

            # Fill in the missing values
            df.loc[df[missing_column].isnull(), missing_column] = predicted_values

            return df
        
        else:
            
            features = X.drop(columns=[missing_column])
            target = X[missing_column]

            features = encoding_categorical_variables(features)
            # Ensure sklearn receives a numeric matrix (no object/string dtypes).
            features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X_train = features.loc[target.notnull()]
            y_train = target.loc[target.notnull()]
            X_predict = features.loc[target.isnull()]

            mlp_estimator = MLPClassifier(
                random_state=42, 
                solver='adam', 
                max_iter=100, 
                hidden_layer_sizes=(100, 50), # Example architecture
                early_stopping=True
            )

            # # Split the data into training and prediction sets
            # train_data = X[X[missing_column].notnull()]
            # predict_data = X[X[missing_column].isnull()]

            if len(X_predict) == 0:
                return df

            if len(X_train) == 0:
                return df

            X_train_np = X_train.to_numpy(dtype=np.float64, copy=True)
            X_predict_np = X_predict.to_numpy(dtype=np.float64, copy=True)

            # Encode class labels to numeric to avoid object-dtype issues.
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train.astype(str))

            if len(label_encoder.classes_) < 2:
                predicted_values = np.repeat(label_encoder.classes_[0], len(X_predict_np))
            else:
                # Fit the model and predict missing values
                mlp_estimator.fit(X_train_np, y_train_encoded)
                predicted_encoded = mlp_estimator.predict(X_predict_np).astype(int)
                predicted_values = label_encoder.inverse_transform(predicted_encoded)

            # X_train = train_data.drop(columns=[missing_column])
            # y_train = train_data[missing_column]

            # X_predict = predict_data.drop(columns=[missing_column])

            # cat_cols = list(X.select_dtypes(include=['object', 'bool']).columns)
            # num_cols = list(X.select_dtypes(exclude=['object', 'bool']).columns)

            # encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            # encoder.fit(X_train[cat_cols])

            # X_train = encoding_categorical_variables(X_train)   
            # X_predict = encoding_categorical_variables(X_predict)

            # Fill in the missing values
            df.loc[df[missing_column].isnull(), missing_column] = predicted_values

            return df

# ===================================================================

def impute_missing_column(df, method, missing_column):
    np.random.seed(0)
    # pandas 3.0+ uses StringDtype for string columns instead of object dtype.
    # Normalize to object so all dtype checks (e.g. dtype == "object") work correctly.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
    imputated_df = pd.DataFrame()
    if method == "no_impute":
        imputator = no_impute()
        imputated_df = imputator.fit(df)
    elif method == "impute_standard":
        imputator = impute_standard()
        imputated_df = imputator.fit(df)
    elif method == "impute_mean":
        imputator = impute_mean()
        imputated_df = imputator.fit_mode(df)
    elif method == "impute_mode":
        imputator = impute_mode()
        imputated_df = imputator.fit(df)
    elif method == "impute_median":
        imputator = impute_median()
        imputated_df = imputator.fit_mode(df)
    elif method == "impute_random":
        imputator = impute_random()
        imputated_df = imputator.fit(df)
    elif method == "impute_knn":
        imputator = impute_knn()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_mice":
        imputator = impute_mice()
        if df[missing_column].dtype in ["float64","int64"]:
            imputated_df = imputator.fit(df, missing_column, estimator=BayesianRidge())
        else:
            imputated_df = imputator.fit(df, missing_column, estimator=KNeighborsClassifier())
    elif method == "impute_linear_regression":
        imputator = impute_linear_regression()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_logistic_regression":
        imputator = impute_logistic_regression()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_random_forest":
        imputator = impute_random_forest()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_cmeans":
        imputator = impute_clustering()
        imputated_df = imputator.fit_num(df, missing_column)
    elif method == "impute_kproto":
        imputator = impute_clustering()
        imputated_df = imputator.fit_cat(df, missing_column)
    # ===================================================================
    elif method == "impute_expectation_maximization":
        imputator = impute_expectation_maximization()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_soft_imputer":
        # print("Using soft imputer===============================================================")
        imputator = impute_soft_imputer()
        imputated_df = imputator.fit(df)
    elif method == "impute_xgb_imputer":
        imputator = impute_xgb_imputer()
        categorical_features = list(df.select_dtypes(include=["object", "bool"]).columns)
        categorical_features_index = [df.columns.get_loc(col) for col in categorical_features]
        imputated_df = imputator.fit(df, missing_column, categorical_features_index, replace_values_back=True)
    elif method == "impute_catboost":
        imputator = impute_catboost()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_rfi":
        imputator = impute_rfi()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_autoimpute":
        imputator = impute_autoimpute()
        imputated_df = imputator.fit(df)
    elif method == "impute_gain":
        imputator = impute_gain()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_vae":
        imputator = impute_vae()
        imputated_df = imputator.fit(df, missing_column)
    elif method == "impute_mlp":
        imputator = impute_mlp()
        imputated_df = imputator.fit(df, missing_column)
    return imputated_df

    
