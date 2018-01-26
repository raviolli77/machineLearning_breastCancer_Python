#!/usr/bin/env python3
import sys
import pandas as pd
from sklearn.externals import joblib
from urllib.request import urlopen
from io import StringIO

sys.path.insert(0, '../src/python/')
from helper_functions import test_set
import random_forest as rf
import knn 
import neural_networks as nn
sys.path.pop(0)

fpr, tpr, auc_knn, _, _ = knn.return_knn()
fpr2, tpr2, auc_rf, _, _ = rf.return_rf()
fpr3, tpr3, auc_nn, _, _ = nn.return_nn()

def clean_crosstab(cross_tab):
   cross_tab_new = cross_tab
   cross_tab_new.reset_index(level=0, inplace=True)
   cross_tab_new.rename(columns ={'index': 'Actual (rows) Vs. Predicted (columns)'}, inplace=True)
   return cross_tab_new

cross_tab_knn = clean_crosstab(knn.test_crosstb)

cross_tab_rf = clean_crosstab(rf.test_crosstb)

cross_tab_nn = clean_crosstab(nn.test_crosstb)

# Classification Report Stuff

def create_class_report(class_report_string):
   class_report_mod = StringIO(class_report_string)
   class_report = pd.read_csv(class_report_mod, ',')
   return class_report


class_rep_knn_str = """
Class,   Precision,  Recall,  F1-score,   Support
Benign,  0.96, 0.93, 0.94, 73
Malignant,  0.88, 0.93, 0.90, 41
Avg/Total,  0.93, 0.93, 0.93, 114
"""

class_rep_knn = create_class_report(class_rep_knn_str)

class_rep_rf_str = """
Class,   Precision,  Recall,  F1-score,   Support
Benign,  0.99, 0.96, 0.97, 73
Malignant,  0.93, 0.98, 0.95, 41
Avg/Total,  0.97, 0.96, 0.97, 114
"""

class_rep_rf = create_class_report(class_rep_rf_str)

class_rep_nn_str = """
Class,   Precision,  Recall,  F1-score,   Support
Benign , 0.99, 0.97, 0.98, 73
Malignant,  0.95, 0.98, 0.96, 41
Avg/Total,  0.97, 0.97, 0.97, 114
"""

class_rep_nn = create_class_report(class_rep_nn_str)

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
