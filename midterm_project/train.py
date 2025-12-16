"""
0. IMPORTS.
"""

import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

"""
0.1. EXPORATORY DATA ANALYSIS.
"""

categorical_variables = ['chestpaintype', 'sex', 'fastingbs', 'restingecg', 'exerciseangina', 'st_slope']
numerical_variables = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']

def load_data():
    df = pd.read_csv('heart.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['fastingbs'] = df['fastingbs'].map({0: 'no', 1: 'yes'})
    df['exerciseangina'] = df['exerciseangina'].map({'N': 'no', 'Y': 'yes'})

    categorical_variables = ['chestpaintype', 'sex', 'fastingbs', 'restingecg', 'exerciseangina', 'st_slope']
    numerical_variables = ['age', 'restingbp', 'cholesterol', 'maxhr', 'oldpeak']

    for categorical_variable in categorical_variables:
        df[categorical_variable] = df[categorical_variable].str.lower().str.replace(' ', '_')

    df[numerical_variables] = df[numerical_variables].fillna(0.0)

    return df

"""
1. SPLITTING THE DATASET USING train_test_split().
"""

def dataframe_split(df):
    train_full_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)
    train_df, validation_df = train_test_split(train_full_df, test_size = 0.25, random_state = 42)

    train_df = train_df.reset_index(drop = True)
    validation_df = validation_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    return train_full_df, train_df, validation_df, test_df

"""
2. FEATURE IMPORTANCE: MUTUAL INFORMATION.
"""

df = load_data()
train_full_df, train_df, validation_df, test_df = dataframe_split(df)

# failure_rate = round(train_df.heartdisease.value_counts(normalize = True), 2)

def mutual_information_heart(series):
    return mutual_info_score(series, train_full_df.heartdisease)

def mutual_information_dataframe():
    mi_score = train_full_df[categorical_variables].apply(mutual_information_heart)
    sorted_scores = mi_score.sort_values(ascending = False)

    cols = ['mi_scores']
    mi_scores_df = pd.DataFrame(sorted_scores, columns = cols)

    return mi_scores_df

"""
2.2. FEATURE IMPORTANCE: CORRELATION.
"""

def correlation_dataframe(train_full_df):
    correlation_score = train_full_df[numerical_variables].corrwith(train_full_df.heartdisease)
    sorted_correlation = correlation_score.sort_values(ascending = False)

    cols = ['correlation_scores']
    correlation_df = pd.DataFrame(sorted_correlation, columns = cols)

    return correlation_df

def create_matrices():
    y_train = train_df.heartdisease.values
    y_validation = validation_df.heartdisease.values
    y_test = test_df.heartdisease.values

    return y_train, y_validation, y_test

y_train, y_validation, y_test = create_matrices()

del train_df['heartdisease']
del validation_df['heartdisease']
del test_df['heartdisease']

"""
3. FEATURE MATRIX.
"""

train_dict = train_df.to_dict(orient = 'records')

dictVectorizer = DictVectorizer(sparse = False)
dictVectorizer.fit(train_dict)
X_train = dictVectorizer.transform(train_dict)

feature_matrix = dictVectorizer.feature_names_

validation_dict = validation_df.to_dict(orient = 'records')
X_validation = dictVectorizer.transform(validation_dict)

test_dict = test_df.to_dict(orient = 'records')
X_test = dictVectorizer.transform(test_dict)

"""
4. LOGISTIC REGRESSION.
* Baseline model.
"""

logistic_model = LogisticRegression(max_iter = 1000, random_state = 42) # type: ignore
logistic_model.fit(X_train, y_train)

y_prediction = logistic_model.predict_proba(X_validation)[:, 1]
logistic_baseline_auc = round(roc_auc_score(y_validation, y_prediction), 4)

"""
4.1. LOGISTIC REGRESSION: PARAMETER TUNING.
* C.
* Penalty = l1.
"""

def logistic_penalty_l1(X_train, y_train):
    c_params = [0.01, 0.1, 1, 10, 100]
    auc_scores = []

    for c_param in c_params:
        logistic_model = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = c_param, max_iter = 1000, random_state = 42) # type: ignore
        logistic_model.fit(X_train, y_train)

        y_prediction = logistic_model.predict_proba(X_validation)[:, 1]
        auc = round(roc_auc_score(y_validation, y_prediction), 4)
        auc_scores.append((c_param, auc))

    cols = ['c_parameters', 'auc_scores']
    logistic_df = pd.DataFrame(auc_scores, columns = cols).sort_values(by = "auc_scores", ascending = False)
    logistic_df.reset_index(drop = True)

    return logistic_df

"""
4.2. LOGISTIC REGRESSION: PARAMETER TUNING.
* C.
* Penalty = l2.
"""

def logistic_penalty_l2(X_train, y_train):
    c_params = [0.01, 0.1, 1, 10, 100]
    auc_scores = []

    for c_param in c_params:
        logistic_model = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = c_param, max_iter = 1000, random_state = 42) # type: ignore
        logistic_model.fit(X_train, y_train)

        y_prediction = logistic_model.predict_proba(X_validation)[:, 1]
        auc = round(roc_auc_score(y_validation, y_prediction), 4)
        auc_scores.append((c_param, auc))

    cols = ['c_parameters', 'auc_scores']
    logistic_df = pd.DataFrame(auc_scores, columns = cols).sort_values(by = "auc_scores", ascending = False)
    logistic_df.reset_index(drop = True)

    return logistic_df

"""
4.3. LOGISTIC REGRESSION: FINAL MODEL.
* C = {0.1}.
* Penalty = l2.
"""

def logistic_model():
    logistic_model = LogisticRegression(penalty = 'l2', solver = 'liblinear', C = 0.1, max_iter = 1000, random_state = 42)
    logistic_model.fit(X_train, y_train)

    y_prediction = logistic_model.predict_proba(X_validation)[:, 1]
    
    return round(roc_auc_score(y_validation, y_prediction), 4)

"""
5. DECISION TREE TRAINING. 
* BASELINE MODEL.
"""

def decision_tree_baseline(X_train, y_train):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(X_train, y_train)

    dt_prediction = tree_model.predict_proba(X_validation)[:, 1] # type: ignore
    
    return round(roc_auc_score(y_validation, dt_prediction), 3)

"""
5.1. DECISION TREE TRAINING: PARAMETER TUNING. 
* max_depth.
* min_samples_leaf.
"""

def decision_tree_depth_tuning(X_train, y_train):
    max_depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, 100, 200, 500]

    for max_depth in max_depths:
        tree_model = DecisionTreeClassifier(max_depth = max_depth)
        tree_model.fit(X_train, y_train)

        dt_prediction = tree_model.predict_proba(X_validation)[:, 1] # type: ignore
        auc_score = roc_auc_score(y_validation, dt_prediction)

        print('%4s => %.3f' % (max_depth, auc_score))

"""
5.2. DECISION TREE TRAINING: PARAMETER TUNING. 
* max_depth ={3, 4, 5}.
* min_samples_leaf.
"""

def decision_tree_samples_tuning(X_train, y_train):
    max_depths = [3, 4, 5]
    min_samples = [1, 2, 3, 4, 5, 6, 10, 15, 20, 100]
    auc_scores = []

    for max_depth in max_depths:
        for min_sample in min_samples:
            tree_model = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_sample)
            tree_model.fit(X_train, y_train)

            dt_prediction = tree_model.predict_proba(X_validation)[:, 1] # type: ignore
            auc_score = roc_auc_score(y_validation, dt_prediction)
            auc_scores.append((max_depth, min_sample, auc_score))
    
    cols = ['max_depth', 'min_sample', 'auc_score']
    decision_tree_df = pd.DataFrame(auc_scores, columns = cols)
    
    return decision_tree_df

"""
5.3. DECISION TREE TRAINING: FINAL MODEL. 
* max_depth = {3}.
* min_samples_leaf = {3}.
"""

def decision_tree_model(X_train, y_train):
    tree_model = DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 3)
    tree_model.fit(X_train, y_train)

    dt_prediction = tree_model.predict_proba(X_validation)[:, 1] # type: ignore
    return round(roc_auc_score(y_validation, dt_prediction), 3)

"""
6. RANDOM FOREST TREE.
* Baseline model.
"""

def random_forest_baseline(X_train, y_train):
    forest_model = RandomForestClassifier(random_state = 42)
    forest_model.fit(X_train, y_train)

    rf_prediction = forest_model.predict_proba(X_validation)[:, 1]

    return round(roc_auc_score(y_validation, rf_prediction), 3)

"""
6.1. RANDOM FOREST TREE: PARAMETER TUNING.
* n_estimators.
* max_depths.
"""

def random_forest_estimators_tuning(X_train, y_train):
    n_estimators = [n for n in range(10, 201, 10)]
    auc_scores = []

    for n_estimator in n_estimators:
        forest_model = RandomForestClassifier(n_estimators = n_estimator, random_state = 42)
        forest_model.fit(X_train, y_train)

        rf_prediction = forest_model.predict_proba(X_validation)[:, 1]
        auc_score = round(roc_auc_score(y_validation, rf_prediction), 3)
        auc_scores.append((n_estimator, auc_score))

    cols = ['n_estimators', 'auc_scores']
    forest_df = pd.DataFrame(auc_scores, columns = cols)

    return forest_df

"""
6.2. RANDOM FOREST TREE: PARAMETER TUNING.
* n_estimators = {170, 190, 200}.
* max_depths.
"""

def random_forest_depth_tuning(X_train, y_train):
    n_estimators = [170, 190, 200] 
    max_depths = [5, 10, 15, 20]
    auc_scores = []

    for max_depth in max_depths:
        for n_estimator in n_estimators:
            forest_model = RandomForestClassifier(n_estimators = n_estimator, max_depth = max_depth, random_state = 42)
            forest_model.fit(X_train, y_train)
        
            rf_prediction = forest_model.predict_proba(X_validation)[:, 1]
            auc_score = round(roc_auc_score(y_validation, rf_prediction), 3)
            auc_scores.append((max_depth, n_estimator, auc_score))

    cols = ['max_depths', 'n_estimators', 'auc_scores']
    forest_df = pd.DataFrame(auc_scores, columns = cols)
    
    return forest_df

"""
6.3. RANDOM FOREST TREE: FINAL MODEL.
* n_estimators = {200}.
* max_depth = {10}.
"""

def random_forest_model(X_train, y_train):
    forest_model = RandomForestClassifier(n_estimators = 200, max_depth = 10, random_state = 42)
    forest_model.fit(X_train, y_train)

    rf_prediction = forest_model.predict_proba(X_validation)[:, 1]
    
    return round(roc_auc_score(y_validation, rf_prediction), 3)

"""
7. XGBOOST TRAINING.
* Baseline model.
"""

dTrain = xgb.DMatrix(X_train, label = y_train, feature_names = feature_matrix)
dValidation = xgb.DMatrix(X_validation, label = y_validation, feature_names = feature_matrix)
watchlist = [(dTrain, 'train'), (dValidation, 'validation')]

def xgb_baseline(dTrain, dValidation, watchlist):
    xgb_params = {
        'eta': 0.3,
        'max_depth': 6,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1
    }

    xgb_model = xgb.train(xgb_params, dTrain, evals = watchlist, num_boost_round = 20, verbose_eval = 5)
    xgb_prediction = xgb_model.predict(dValidation)

    return round(roc_auc_score(y_validation, xgb_prediction), 3)

"""
7.1. XGBOOST: PARAMETER TUNING.
* eta.
* max_depth.
* min_child_weight.
"""

def xgb_eta_tuning(dTrain, watchlist):
    xgb_params = [{'eta': 0.01, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.1, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.2, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'eval_metric': 'auc', 'objective': 'binary:logistic'}]

    for xgb_param in xgb_params:
        xgb_model = xgb.train(xgb_param, dTrain, evals = watchlist, num_boost_round = 20, verbose_eval = 5)

"""
7.2. XGBOOST: PARAMETER TUNING.
* eta = {0.3}.
* max_depth.
* min_child_weight.
"""

def xgb_depth_tuning(dTrain, watchlist):
    xgb_params = [{'eta': 0.3, 'max_depth': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'max_depth': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'max_depth': 3, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'max_depth': 6, 'eval_metric': 'auc', 'objective': 'binary:logistic'}]

    for xgb_param in xgb_params:
        xgb_model = xgb.train(xgb_param, dTrain, evals = watchlist, num_boost_round = 20, verbose_eval = 5)

"""
7.3. XGBOOST: PARAMETER TUNING.
* eta = {0.3}.
* max_depth = {3}.
* min_child_weight.
"""

def xgb_weight_tuning(dTrain, watchlist):
    xgb_params = [{'eta': 0.3, 'max_depth': 3, 'min_child_weight': 1, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'max_depth': 3, 'min_child_weight': 2, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'max_depth': 3, 'min_child_weight': 3, 'eval_metric': 'auc', 'objective': 'binary:logistic'},
                  {'eta': 0.3, 'max_depth': 3, 'min_child_weight': 6, 'eval_metric': 'auc', 'objective': 'binary:logistic'}]

    for xgb_param in xgb_params:
        xgb_model = xgb.train(xgb_param, dTrain, evals = watchlist, num_boost_round = 20, verbose_eval = 5)

"""
7.4. XGBOOST: FINAL MODEL.
* eta = {0.3}.
* max_depth = {3}.
* min_child_weight = {1}.
"""

xgb_params = {
    'eta': 0.3,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

xgb_model = xgb.train(xgb_params, dTrain, evals = watchlist, num_boost_round = 16, verbose_eval = 5)
xgb_prediction = xgb_model.predict(dValidation)

xgb_auc = round(roc_auc_score(y_validation, xgb_prediction), 3)
print("")

"""
8. FINAL MODEL
* XGBoost.
* On the test set.
"""

dTest = xgb.DMatrix(X_test, label = y_test, feature_names = feature_matrix)

xgb_params = {
    'eta': 0.3,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1
}

watchlist = [(dTrain, 'train'), (dTest, 'test')]

model = xgb.train(xgb_params, dTrain, evals = watchlist, num_boost_round = 16, verbose_eval = 5)
model_prediction = model.predict(dTest)

model_auc = round(roc_auc_score(y_test, model_prediction), 3)

"""
9. SAVING THE MODEL USING PICKLE.
"""

output_file = f'heartfailure_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dictVectorizer, model, feature_matrix), f_out)

print("Model successfully saved.")