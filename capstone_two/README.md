## LUNG CANCER PREDICTION IN PATIENTS
### 1. PROJECT DESCRIPTION
This project trains and evaluates models in predicting lung cancer likelihood in patients using a lung cancer dataset.

### 2. DATASET
The lung cancer dataset has 309 records, with 12 features. **age** is the only numerical variable in the dataset, the rest are categorical variables. Some features are features that indicate symptoms whilst some features are features that indicate behavioural traits of a patient.
The features are:
* **gender**: [M or F]
* **age**: age of the patient.
* **smoking**: indicates whether the patient is a smoker or not, [1: No, 2: Yes].
* **yellow fingers**: indicates whether the patient has yellow fingers, [1: No, 2: Yes].
* **anxiety**: indicates whether the patient suffers from anxiety, [1: No, 2: Yes].
* **peer pressure**: indicates whether the patient suffers from peer pressure, [1: No, 2: Yes].
* **chronic disease**: indicates whether the patient has a chronic disease, [1: No, 2: Yes].
* **fatigue**: indicates whether the patient suffers from fatigue, [1: No, 2: Yes].
* **allergy**: indicates whether the patient has any allergies, [1: No, 2: Yes].
* **wheezing**: indicates whether the patient makes any high pitched sounds when breathing or not, [1: No, 2: Yes].
* **alcohol consuming**: indicates whether the patient drinks alcohol, [1: No, 2: Yes].
* **coughing**: indicates whether the patient has any coughing symptoms, [1: No, 2: Yes].
* **shortness of breath**: indicates whether the patient suffers from shortness of breath, [1: No, 2: Yes].
* **swallowing difficulty**: indicates whether the patient has any pains when swallowing, [1: No, 2: Yes].
* **chest pain**: indicates whether the patient has any chest pains, [1: No, 2: Yes].
* **lung cancer**: the target variable, indicates presence of cancer in the patient, [0: No, 1: Yes]. 
  
### 3. EXPLORATORY DATA ANALYSIS
In this part:
* column headers are converted to lowercase alphabets and whitespaces are replaced by underscores.
* features are classified into 2 classes: **categorical_variables** and **numerical_variables**.
* attributes in the dataset are then also converted to lowercase alphabets, and whitespaces replaced by underscores.
* the dataset is then checked for missing values.
* check for duplicated rows, dataset is trimmed down to 276 rows.
* mapping of feature values, {1 => 0: No, 2 => 1: Yes}.
* check whether the target variables' values are balanced.

### 4. SPLITTING THE DATASET
The dataset is split into 2 dataframes: **train_df**(80%) and **test_df**(20%) using scikit-learn's **train_test_split()**.
* **stratify()** is used to ensure that the proportions of classes in the target variable are preserved in the 2 new sets.
  
### 5. FEATURE IMPORTANCE
* check the proportions of classes in the target variable of the 2 sets.
* mutual information and correlation are evaluated.
* mutual importance evaluates the importance of categorical variables to the target variable.
* correlation evaluates the importance of numerical variables to the target variable.

### 6. FEATURE MATRIX
In this part:
* the target variable is extracted from the 2 dataframes **train_df** and **test_df**.
* the two dataframes are transformed into 2 dictionaries: **train_dict** and **test_dict**.
* the 2 dictionaries are then converted to 2 matrices: **X_train** and **X_test**.
* a feature matrix is extracted from the **dictVectorizer()**.
  
### 7. MODELS EVALUATED
For each model, I have evaluated it's baseline model and then moved on to hyperparameter tuning in an attempt to improve each model's performance. At the end of each section, I have provided a tuned model that has the highest **roc_auc_score** as that section's **final model.** 
* **Logistic Regression**: trained using a resampled set because of the imbalanced nature of the dataset.
* **Decision Tree**
* **Random Forest Tree** 
* **XGBoost**
* **Neural Networks**
  
### 8. FINAL MODEL
The best-performing model is selected to be the XGBoost model with the following hyperparameters:
* eta = {0.2}.
* max_depth = {3}.
* min_child_weight = {2}.
  
### 9. SAVING & LOADING THE MODEL
In this part:
* the module **pickle** is used to save and load the model.
* the model is saved as **model.bin**.
  
### 10. PYTHON SCRIPTS
In this part:
* **notebook.ipynb** is converted into a python script and then split into 2 files:
* **train.py** contains code that covers everything from the **EDA** step to the **saving & loading the model** step.
* **predict.py** contains code for users to access & test the model through web service developed using the **FastAPI** module.
  
### 11. TESTING THE MODEL
* run the following command in the terminal to install the required packages.
```bash
pip install fastapi imbalanced-learn matplotlib numpy pandas requests scikit-learn tensorflow uv uvicorn xgboost
```

* download the lung cancer dataset from kaggle:
```bash
curl -O "https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer"
```

* run the **train.py** script, this will save the **model.bin** in the same folder.
```bash
python3 train.py
```

* to run the web service:
```bash
fastapi dev predict.py --port 8000
```

OR

```bash
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
```

Then open the API documentation and send a request to test the model:

```bash
{
  "gender": "m",
  "age": 29,
  "smoking": "no",
  "yellow_fingers": "no",
  "anxiety": "yes",
  "peer_pressure": "yes",
  "chronic_disease": "no",
  "fatigue": "no",
  "allergy": "no",
  "wheezing": "yes",
  "alcohol_consuming": "yes",
  "coughing": "yes",
  "shortness_of_breath": "yes",
  "swallowing_difficulty": "yes",
  "chest_pain": "yes"
}
```
Output should be:
```bash
{
  "lung cancer probability": 0.22855529189109802,
  "lung cancer": false
}
```

### 12. RUN MODEL USING DOCKER
```bash
docker run --rm capstone_two
```

OR

```bash
docker run -p 8000:8000 capstone_two
```

* **capstone_two** is the name of the docker image.
* this should return output that contains this line:

```bash
Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
