import argparse

# Data handling and visaluzation
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# SVM
from sklearn import svm

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Model evaluation
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, classification_report, accuracy_score

# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Save model
import joblib

# Hyperparameters search
from sklearn.model_selection import GridSearchCV

import logging

logging.info("Starting...")

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="Path to data", default="", type=str, required=True)
parser.add_argument("-s", "--scale", help="Type of scaling", choices=['None', 'MinMax', 'Standard'], default="None")
parser.add_argument("-k", "--kernel", help="Kernel function", choices=["polynomial", "rbf", "None"], default="None")
args = parser.parse_args()

# ## Setup

logging.info("Reading data...")
data = pd.read_csv(args.data)
data.columns = ["Age(years)", "Gender", "Height(cm)", "Weight(kg)", "SystolicPressure", "DiastolicPressure", "Cholesterol", "Glucose", "Smoke", "Alcohol", "Active", "cardio_disease", "BMI", "Pulse"]
cardio_disease = data.loc[:,"cardio_disease"].map({"No":0, "Yes":1})
data = data.loc[:, ["Age(years)", "Weight(kg)", "SystolicPressure", "DiastolicPressure", "Cholesterol", "Glucose", "Smoke", "Alcohol", "Active", "BMI", "Pulse"]]
logging.info(data.shape)
data.head()


# Transform data to numerical values
data.Cholesterol = data.Cholesterol.map({"Normal":1, "Above-Normal":2, "Well-Above-Normal":3})
data.Glucose = data.Glucose.map({"Normal":1, "Above-Normal":2, "Well-Above-Normal":3})
data.Smoke = data.Smoke.map({"No":0, "Yes":1})
data.Alcohol = data.Alcohol.map({"No":0, "Yes":1})
data.Active = data.Active.map({"No":0, "Yes":1})
data.head()

if args.scale == "MinMax":
    logging.info("Scaling data to [0,1] range")
    scaler = MinMaxScaler().fit(data)
    data = scaler.fit_transform(data) 

elif args.scale == "Standard":
    logging.info("Standardizing data...")
    std_scaler = StandardScaler().fit(data)
    data = std_scaler.fit_transform(data)
elif args.scale == "None":
    data = data.values

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(data, cardio_disease, test_size = 0.2, random_state = 42)
logging.info('Training Features Shape:', train_features.shape)
logging.info('Training Labels Shape:', train_labels.shape)
logging.info('Testing Features Shape:', test_features.shape)
logging.info('Testing Labels Shape:', test_labels.shape)


logging.info("Starting training...")
if args.kernel == "rbf":
    logging.info("Starting Grid search for rbf kernel")
    # Defining parameter range 
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf']} 

    grid = GridSearchCV(svm.SVC(), param_grid, verbose = 3) 

    # fitting the model for grid search 
    grid.fit(train_features, train_labels) 

elif args.kernel == "polynomial":
    logging.info("Starting Grid search for polynomial kernel")
    # Defining parameter range 
    param_grid = {'degree': [2, 3],
                  "coef0": [0, 0.01, 0.1, 1, 10],
                'kernel': ['poly']} 

    grid = GridSearchCV(svm.SVC(), param_grid, verbose = 3) 

    # fitting the model for grid search 
    grid.fit(train_features, train_labels) 

elif args.kernel == "None":
    grid = svm.SVC(kernel="linear", C=1.0, random_state=42)
    grid.fit(train_features, train_labels)
    



# save
logging.info("Saving model...")
joblib.dump(grid, f"{args.scale}_scale_{args.kernel}_kernel_svm_model.pkl") 

