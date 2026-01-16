"""
0. IMPORTS
"""
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import tensorflow as tf

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings("ignore", category = RuntimeWarning)
warnings.filterwarnings("ignore", category = UserWarning)

"""
1. EXPLORATORY DATA AANALYSIS
"""
df = pd.read_csv("lung_cancer.csv")
output_file = f'model.bin'

def data_preprocessing(df):
    # check for duplicated rows and drop them
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    df = df.drop_duplicates()
    df = df.reset_index(drop = True)

    # create class columns, then map the binary columns {1: no, 2: yes}
    binary_variables = ['smoking', 'yellow_fingers', 'anxiety', 'peer_pressure',
                    'chronic_disease', 'fatigue', 'allergy', 'wheezing',
                    'alcohol_consuming', 'coughing', 'shortness_of_breath',
                    'swallowing_difficulty', 'chest_pain']

    for binary_variable in binary_variables:
        df[binary_variable] = df[binary_variable].map({1: 0, 2: 1})

    for binary_variable in binary_variables:
        df[binary_variable] = df[binary_variable].map({0: 'no', 1: "yes"})

    df.lung_cancer = df.lung_cancer.map({'NO': 0, 'YES': 1})

    numerical_variables = ['age']
    categorical_variables = ['gender'] + binary_variables

    for categorical_variable in categorical_variables:
        df[categorical_variable] = df[categorical_variable].str.lower().str.replace(' ', '_')
    
    # return a tuple
    return categorical_variables, df, numerical_variables

categorical_variables, df, numerical_variables = data_preprocessing(df)

"""
1. SPLITTING DATASET USING train_test_split()
* 80/20 split.
"""
def dataset_split(df):
    # target variable
    y = df['lung_cancer']

    # stratify ensures that the proportions of classes in the target variable are preserved.
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42, stratify = y)
    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    return train_df, test_df

train_df, test_df = dataset_split(df)

"""
2. FEATURE IMPORTANCE
* mutual importance (between categorical variables).
* correlation (between numerical variables).
"""
def target_distributions(train_df, test_df):
    # check distribution of the target variable in the datasets
    train_df['lung_cancer'].value_counts(normalize = True)
    test_df['lung_cancer'].value_counts(normalize = True)

def lungs_mutual_information(series):
    return mutual_info_score(series, train_df.lung_cancer)

def mutual_information_df(train_df):
    mi_score = train_df[categorical_variables].apply(lungs_mutual_information).sort_values(ascending = False)
    
    cols = ['mi_scores']
    return pd.DataFrame(mi_score, columns = cols).sort_values(by = "mi_scores").reset_index(drop = True)

def correlation_df(train_df):
    correlation_score = train_df[numerical_variables].corrwith(train_df.lung_cancer).sort_values(ascending = False)

    cols = ['correlation_score']
    return pd.DataFrame(correlation_score, columns = cols).sort_values(by = "correlation_scores").reset_index(drop = True)

"""
3. FEATURE MATRIX. 
* using a DictVectorizer.
* one-hot encoding.
"""

# extract target variables from dataframes and store them as matrices:
y_train = train_df.lung_cancer.values
y_test = test_df.lung_cancer.values

# remove the target variable from dataframes:
del train_df['lung_cancer']
del test_df['lung_cancer']

def dataframe_transormation(train_df, test_df):
    # transform dataframes to dictionaries:
    train_dict = train_df.to_dict(orient = 'records')
    test_dict = test_df.to_dict(orient = 'records')

    # initialise dictVectorizer and fit the training dictionary into it:
    dictVectorizer = DictVectorizer(sparse = False)
    dictVectorizer.fit(train_dict)

    # extract dictVectorizer's feature matrix:
    feature_matrix = dictVectorizer.feature_names_

    # transform dictionaries to matrices:
    X_train = dictVectorizer.transform(train_dict)
    X_test = dictVectorizer.transform(test_dict)

    return dictVectorizer, feature_matrix, X_test, X_train

dictVectorizer, feature_matrix, X_test, X_train = dataframe_transormation(train_df, test_df)

def resampled_data(X_train, y_train):
    smote = SMOTE(random_state = 42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) # type: ignore

    return X_train_resampled, y_train_resampled

X_train_resampled, y_train_resampled = resampled_data(X_train, y_train)

"""
4. LOGISTIC REGRESSION MODEL TRAINING.
"""
def baseline_regression(X_train_resampled, X_test, y_train_resampled):
    lr_model = LogisticRegression(max_iter = 1000, random_state = 42)
    lr_model.fit(X_train_resampled, y_train_resampled)
    
    y_prediction = lr_model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, y_prediction)

"""
4.1. LOGISTIC REGRESSION: HYPERPARAMETER TUNING.
"""
def regression_l1_tuning(X_train_resampled, X_test, y_train_resampled):
    c_params = [0.01, 0.1, 1, 10, 100]
    auc_scores = []

    for c_param in c_params:
        logistic_model = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = c_param, max_iter = 1000, random_state = 42)
        logistic_model.fit(X_train_resampled, y_train_resampled)

        y_prediction = logistic_model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, y_prediction), 4)
        auc_scores.append((c_param, auc))

    cols = ['c_parameters', 'auc_scores']
    return pd.DataFrame(auc_scores, columns = cols).sort_values(by = "auc_scores", ascending = False).reset_index(drop = True)

"""
4.2. LOGISTIC REGRESSION: HYPERPARAMETER TUNING.
"""
def regression_l2_tuning(X_train_resampled, X_test, y_train_resampled):
    c_params = [0.01, 0.1, 1, 10, 100]
    auc_scores = []

    for c_param in c_params:
        logistic_model = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = c_param, max_iter = 1000, random_state = 42)
        logistic_model.fit(X_train_resampled, y_train_resampled)

        y_prediction = logistic_model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, y_prediction), 4)
        auc_scores.append((c_param, auc))

    cols = ['c_parameters', 'auc_scores']
    return pd.DataFrame(auc_scores, columns = cols).sort_values(by = "auc_scores", ascending = False).reset_index(drop = True)

"""
4.3. LOGISTIC REGRESSION: FINAL MODEL
"""
def regression_final_model(X_train_resampled, y_train_resampled):
    regression_model = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.01, max_iter = 1000, random_state = 42)
    regression_model.fit(X_train_resampled, y_train_resampled)

    y_prediction = regression_model.predict_proba(X_test)[:, 1]
    return round(roc_auc_score(y_test, y_prediction), 4)

"""
5. DECISION TREE MODEL TRAINING.
"""
def baseline_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    
    dt_prediction = dt_model.predict_proba(X_test)[:, 1] # type: ignore
    return roc_auc_score(y_test, dt_prediction)

"""
5.1. DECISION TREE MODEL: HYPERPARAMETER TUNING
"""
def tree_depths_tuning(X_train, y_train, X_test, y_test):
    max_depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, 100, 200, 500]
    depths = []

    for max_depth in max_depths:
        tree_model = DecisionTreeClassifier(max_depth = max_depth)
        tree_model.fit(X_train, y_train)

        dt_prediction = tree_model.predict_proba(X_test)[:, 1] # type: ignore
        auc_score = roc_auc_score(y_test, dt_prediction)
        
        depths.append((max_depth, auc_score))

    cols = ['max_depths', 'auc_scores']
    return pd.DataFrame(depths, columns = cols).sort_values(by = 'auc_scores', ascending = False).reset_index(drop = True)

"""
5.2. DECISION TREE MODEL: HYPERPARAMETER TUNING
"""
def tree_samples_tuning(X_train, y_train, X_test, y_test):
    max_depths = [3, 4, 5]
    min_samples = [1, 2, 3, 4, 5, 6, 10, 15, 20, 100]
    auc_scores = []

    for max_depth in max_depths:
        for min_sample in min_samples:
            tree_model = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_sample)
            tree_model.fit(X_train, y_train)

            ydt_prediction = tree_model.predict_proba(X_test)[:, 1] # type: ignore
            auc_score = roc_auc_score(y_test, ydt_prediction)
            auc_scores.append((max_depth, min_sample, auc_score))
    
    cols = ['max_depths', 'min_samples', 'auc_scores']
    return pd.DataFrame(auc_scores, columns = cols).sort_values(by = 'auc_scores', ascending = False).reset_index(drop = True)

"""
5.3. DECISION TREE: FINAL MODEL
"""
def tree_final_model(X_train, y_train, X_test, y_test):
    decision_tree_model = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 1,  random_state = 42)
    decision_tree_model.fit(X_train, y_train)

    dt_prediction = decision_tree_model.predict_proba(X_test)[:, 1] # type: ignore
    roc_auc_score(y_test, dt_prediction)

"""
6. RANDOM FOREST TREE MODEL TRAINING
"""
def baseline_forest(X_train, y_train, X_test, y_test):
    forest_model = RandomForestClassifier(random_state = 42)
    forest_model.fit(X_train, y_train)

    rf_prediction = forest_model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, rf_prediction)

"""
6.1. RANDOM FOREST TREE MODEL: HYPERPARAMETER TUNING
"""
def forest_estimators_tuning(X_train, y_train, X_test, y_test):
    n_estimators = [n for n in range(10, 201, 10)]
    auc_scores = []

    for n_estimator in n_estimators:
        forest_model = RandomForestClassifier(n_estimators = n_estimator, random_state = 42)
        forest_model.fit(X_train, y_train)

        rf_prediction = forest_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, rf_prediction)
        auc_scores.append((n_estimator, auc_score))

    cols = ['n_estimators', 'auc_scores']
    return pd.DataFrame(auc_scores, columns = cols).sort_values(by = 'auc_scores', ascending = False).reset_index(drop = True)

"""
6.2. RANDOM FOREST TREE MODEL: HYPERPARAMETER TUNING
"""
def forest_depths_model(X_train, y_train, X_test, y_test):
    n_estimators = [10, 20, 30, 200] 
    max_depths = [5, 10, 15, 20]
    auc_scores = []

    for max_depth in max_depths:
        for n_estimator in n_estimators:
            forest_model = RandomForestClassifier(n_estimators = n_estimator, max_depth = max_depth, random_state = 42)
            forest_model.fit(X_train, y_train)
        
            rf_prediction = forest_model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, rf_prediction)
            auc_scores.append((max_depth, n_estimator, auc_score))

    cols = ['max_depths', 'n_estimators', 'auc_scores']
    return pd.DataFrame(auc_scores, columns = cols).sort_values(by = 'auc_scores', ascending = False).reset_index(drop = True)

"""
6.3. RANDOM FOREST TREE: FINAL MODEL
"""
def forest_final_model(X_train, y_train, X_test, y_test):
    random_forest_model = RandomForestClassifier(n_estimators = 10, max_depth = 10, random_state = 42)
    random_forest_model.fit(X_train, y_train)

    rf_prediction = random_forest_model.predict_proba(X_test)[:, 1]
    roc_auc_score(y_test, rf_prediction)

"""
7. XGBOOST MODEL TRAINING
"""
dTrain = xgb.DMatrix(X_train, label = y_train, feature_names = feature_matrix)
dTest = xgb.DMatrix(X_test, label = y_test, feature_names = feature_matrix)

pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

def baseline_xgb(dTrain, dTest, pos_weight, y_test):
    xgb_params = {
        'eta': 0.3,
        'max_depth': 6,
        'scale_pos_weight': pos_weight,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }

    xgb_model = xgb.train(xgb_params, dTrain, evals = [(dTrain, 'train'), (dTest, 'test')], num_boost_round = 7, verbose_eval = 5)
    
    xgb_prediction = xgb_model.predict(dTest)
    return roc_auc_score(y_test, xgb_prediction)

"""
7.1. XGBOOST: HYPERPARAMETER TUNING
"""
def eta_tuning_xgb(dTrain, dTest, pos_weight):
    xgb_params = [{'eta': 0.01, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.3, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight}]

    for xgb_param in xgb_params:
        xgb.train(xgb_param, dTrain, evals = [(dTrain, 'train'), (dTest, 'test')], num_boost_round = 7, verbose_eval = 5)

"""
7.2. XGBOOST: HYPERPARAMETER TUNING
"""
def depths_tuning_xgb(dTrain, dTest, pos_weight):
    xgb_params = [{'eta': 0.2, 'max_depth': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'max_depth': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'max_depth': 3, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'max_depth': 6, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight}]

    for xgb_param in xgb_params:
        xgb.train(xgb_param, dTrain, evals = [(dTrain, 'train'), (dTest, 'test')], num_boost_round = 7, verbose_eval = 5)

"""
7.3. XGBOOST: HYPERPARAMETER TUNING
"""
def weights_tuning_xgb(dTrain, dTest, pos_weight):
    xgb_params = [{'eta': 0.2, 'max_depth': 3, 'min_child_weight': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'max_depth': 3, 'min_child_weight': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'max_depth': 3, 'min_child_weight': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight},
                  {'eta': 0.2, 'max_depth': 3, 'min_child_weight': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic', 'scale_pos_weight': pos_weight}]

    for xgb_param in xgb_params:
        xgb.train(xgb_param, dTrain, evals = [(dTrain, 'train'), (dTest, 'test')], num_boost_round = 7, verbose_eval = 5)

"""
7.4. XGBOOST: FINAL MODEL
"""
def final_model_xgb(dTrain, dTest, pos_weight, y_test):
    xgb_params = {
        'eta': 0.2,
        'max_depth': 3,
        'min_child_weight': 2,
        'scale_pos_weight': pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }

    xgb_model = xgb.train(xgb_params, dTrain, evals = [(dTrain, 'train'), (dTest, 'test')], num_boost_round = 7, verbose_eval = 5)
    xgb_prediction = xgb_model.predict(dTest)

    return roc_auc_score(y_test, xgb_prediction)

"""
8. NEURAL NETWORK MODEL TRAINING
"""
def nn_model(X_train_resampled, y_train_resampled, X_test, y_test):
    weights = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    class_weight = dict(enumerate(weights))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = tf.keras.Sequential([tf.keras.layers.Dense(units = 64, activation = 'relu'), # type: ignore
                                    tf.keras.layers.Dense(units = 32, activation = 'relu'), # type: ignore
                                    tf.keras.layers.Dense(units = 1, activation = 'sigmoid')]) # type: ignore

    nn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['auc'])
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
    history = nn_model.fit(X_train_scaled, y_train, validation_split = 0.3, epochs = 20, batch_size = 16, class_weight = class_weight, callbacks = [early_stop])

    nn_prediction = nn_model.predict(X_test_scaled)
    # evaluate() returns a tuple [test_loss, test_accuracy]
    test_loss, test_auc = nn_model.evaluate(X_test_scaled, y_test)

    return history, nn_prediction, test_loss, test_auc

"""
9. FINAL MODEL: XGBOOST
"""
def final_model(dTrain, dTest, pos_weight, y_test):
    xgb_params = {
        'eta': 0.2,
        'max_depth': 3,
        'min_child_weight': 2,
        'scale_pos_weight': pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }

    model = xgb.train(xgb_params, dTrain, evals = [(dTrain, 'train'), (dTest, 'test')], num_boost_round = 7, verbose_eval = 5)
    
    y_prediction = model.predict(dTest)
    return roc_auc_score(y_test, y_prediction)

model = final_model(dTrain, dTest, pos_weight, y_test)

"""
10. SAVE MODEL
"""

with open(output_file, 'wb') as f_out:
    pickle.dump((dictVectorizer, model, feature_matrix), f_out)
    print("model successfully saved.")