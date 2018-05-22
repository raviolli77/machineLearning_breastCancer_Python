#!/usr/bin/env python3
import sys
import pandas as pd
from sklearn.externals import joblib
from urllib.request import urlopen
from io import StringIO

# Importing src python scripts ----------------------
sys.path.insert(0, '../src/python/')
from knn import fit_knn
from random_forest import fit_rf
from neural_networks import fit_nn
from data_extraction import test_set_scaled
from data_extraction import test_set, test_class_set
from helper_functions import create_conf_mat
from produce_model_metrics import produce_model_metrics
sys.path.pop(0)

# Calling up metrics from the model scripts
# KNN -----------------------------------------------
metrics_knn = produce_model_metrics(fit_knn, test_set,
                                    test_class_set, 'knn')
# Call each value from dictionary
predictions_knn = metrics_knn['predictions']
accuracy_knn = metrics_knn['accuracy']
fpr = metrics_knn['fpr']
tpr = metrics_knn['tpr']
auc_knn = metrics_knn['auc']

test_error_rate_knn = 1 - accuracy_knn

# Confusion Matrix
cross_tab_knn = create_conf_mat(test_class_set,
                                predictions_knn)

# RF ------------------------------------------------
metrics_rf = produce_model_metrics(fit_rf, test_set,
                                   test_class_set, 'rf')
# Call each value from dictionary
predictions_rf = metrics_rf['predictions']
accuracy_rf = metrics_rf['accuracy']
fpr2 = metrics_rf['fpr']
tpr2 = metrics_rf['tpr']
auc_rf = metrics_rf['auc']

test_error_rate_rf = 1 - accuracy_rf

cross_tab_rf = create_conf_mat(test_class_set,
                               predictions_rf)

# NN ----------------------------------------
metrics_nn = produce_model_metrics(fit_nn, test_set_scaled,
                                   test_class_set, 'nn')

# Call each value from dictionary
predictions_nn = metrics_nn['predictions']
accuracy_nn = metrics_nn['accuracy']
fpr3 = metrics_nn['fpr']
tpr3 = metrics_nn['tpr']
auc_nn = metrics_nn['auc']

test_error_rate_nn = 1 - accuracy_nn

cross_tab_nn = create_conf_mat(test_class_set,
                               predictions_nn)

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
