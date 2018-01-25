#!/usr/bin/env python3
import sys
import pandas as pd
from sklearn.externals import joblib
from urllib.request import urlopen

sys.path.insert(0, '../src/python/')
from helper_functions import test_set
import random_forest as rf
import knn 
import neural_networks as nn
sys.path.pop(0)

fpr, tpr, auc_knn, _, _ = knn.return_knn()
fpr2, tpr2, auc_rf, _, _ = rf.return_rf()
fpr3, tpr3, auc_nn, _, _ = nn.return_nn()

cross_tab_knn = knn.test_crosstb
cross_tab_knn.reset_index(level=0, inplace=True)
cross_tab_knn.rename(columns ={'index': 'Actual (rows) Vs. Predicted (columns)'}, inplace=True)

cross_tab_rf = rf.test_crosstb
cross_tab_rf.reset_index(level=0, inplace=True)
cross_tab_rf.rename(columns ={'index': 'Actual (rows) Vs. Predicted (columns)'}, inplace=True)  

cross_tab_nn = nn.test_crosstb
cross_tab_nn.reset_index(level=0, inplace=True)
cross_tab_nn.rename(columns ={'index': 'Actual (rows) Vs. Predicted (columns)'}, inplace=True)

# Loading data and cleaning dataset
UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean',
         'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean',
         'concavity_mean','concave_points_mean',
         'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se',
         'area_se', 'smoothness_se', 'compactness_se',
         'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se',
         'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst',
         'smoothness_worst', 'compactness_worst',
         'concavity_worst', 'concave_points_worst',
         'symmetry_worst', 'fractal_dimension_worst']

breast_cancer = pd.read_csv(urlopen(UCI_data_URL), names=names)

# Setting 'id_number' as our index
breast_cancer.set_index(['id_number'], inplace = True)

# Converted to binary to help later on with models and plots
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})
