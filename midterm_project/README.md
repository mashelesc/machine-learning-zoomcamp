## 0. Heart Failure Prediction in Patients
### 1. Model Description
In South Africa, cardiovascular disease is the second-highest cause of death behind only to HIV/AIDS. Heart failure contributes significantly to this, so what causes heart failure? 

This project's aim is to answer that question. So by using a dataset that has 12 features, ranging from the patient's age to their exercise angina and their maximum heart rate, this project starts by examining feature importance with the desire to see the factors that contribute more to a positive heart failure likelihood. It then moves on to testing different scikit-learn models, so that it can find a model that more accurately describes the dataset. It eventually ends with an XGBoost model that has an roc score of at least 94%, that is a model that will 94/100 times correctly predict whether a patient is likely to suffer heart failure or not.

### 2. Dataset Features Description
The dataset has 918 records, with 12 features:
* age: age of the patient (in years).
* sex: sex of the patient (m: male, f: female).
* chest pain type: chest pain type (ta: typical angina, ata: atypical angina, nap: non-anginal pain, asy: asymptomatic).
* resting bp: resting blood pressure (mm Hg).
* cholesterol: serum cholesterol (mm/dl).
* fasting bs: fasting blood sugar (1: fasting bs > 120 mg/dl, 0: fasting bs < 120 mg/dl).
* resting ecg: resting electrocardiogram results (normal: normal, st: having ST-T wave abnormality, lvh: showing probable left ventricular hypertrophy by Estes' criteria).
* max hr: maximum heart rate achieved (in the range 60 and 202).
* exercise angina: exercise-induced angina (y: yes, n: no).
* old peak: old peak = extent of ST depression induced by exercise relative to rest.
* st_slope: the slope of the peak exercise ST segment (up: up sloping, flat: flat sloping, down: down sloping).
* heart disease: Target Variable (1: heart disease, 0: normal heart).

### 3. Exploratory Data Analysis
In this part:
* column headers are converted to lowercase alphabets and whitespaces are replaced by underscores.
* features are classified into 2 classes: **categorical_variables** and **numerical_variables**.
* attributes in the dataset are then also converted to lowercase alphabets, and whitespaces replaced by underscores.
* the dataset is then checked for missing values.
* descriptive statistics performed on the dataset using pandas' **.describe()** method.

### 4. Splitting The Dataset
The dataset is split into 3 sets of size 60/20/20 using scikit-learn's **train_test_split()** function.
* 60% training set.
* 20% validation set.
* 20% test set.

### 5. Feature Importance
In this part:
* global rate is calculated to check for class imbalance.
* mutual information: is calculated to measure importance of categorical variables.
* correlation: is calculated to measure importance of numerical variables.

### 6. Feature Matrix
In this part, the **DictVectorizer()** function is used to create 3 matrices from 3 dictionaries.
* the training set is first converted into a dictionary using pandas' **to_dict()** function, and then converted into the matrix **X_train**.
* the validation set is first converted into a dictionary using pandas' **to_dict()** function, and then converted into the matrix **X_validation**.
* the test set is first converted into a dictionary using pandas' **to_dict()** function, and then converted into the matrix **X_test**.
* the feature matrix is then extracted from **dictVectorizer** using **DictVectorizer()**'s **feature_names_** function.  

### 7. Models Evaluated on The Dataset
In this part, models are evaluated and then their performance is compared using their **roc_auc_score**. 
* Logistic Regression: first on the baseline model, and then parameter tuning on the **C** and **solver** parameters.
* Decision Tree: first on the baseline model, and then parameter tuning on the **max_depth** and **min_samples_leaf** parameters.
* Random Forest Tree: first on the baseline model, and then parameter tuning on the **max_depth** and **n_estimators** parameters.
* XGBoost: first on the baseline model, and then parameter tuning on the **eta**, **max_depth**, and **min_child_weight** parameters.

### 8. Final Model
The best-performing model is selected to be the XGBoost model with the following parameters, this model is tested on the test set:
* eta = {0.3}.
* max_depth = {3}.
* min_child_weight = {1}.

### 9. Python Scripts
* **notebook.ipynb** script is exported into **train.py**.
* the final model is saved as **heartfailure_model.bin** using pickle.
* **predict.py** uses **FastAPI** to predict the patient's heart failure probability. 
