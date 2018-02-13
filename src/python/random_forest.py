#!/usr/bin/env python3

#####################################################
##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
#####################################################
#
# Project by Raul Eulogio
#
# Project found at: https://www.inertia7.com/projects/3
#

"""
Random Forest Classification
"""

import time
import sys, os
import numpy as np
import pandas as pd
import helper_functions as hf
import matplotlib.pyplot as plt
from helper_functions import training_set, class_set, test_set, test_class_set
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.externals import joblib

# Fitting Random Forest
# Set the random state for reproducibility
fit_rf = RandomForestClassifier(random_state=42)

## Set best parameters given by grid search
fit_rf.set_params(criterion = 'gini',
                  max_features = 'log2',
                  max_depth = 3,
                  n_estimators=400,
                  oob_score=True)

# Training Model
fit_rf.fit(training_set,
	class_set)

# Extracting feature importance
importances_rf = fit_rf.feature_importances_

# Create indices for importance of features
indices_rf = np.argsort(importances_rf)[::-1]

# Random Forest Predictions on Test Set
predictions_rf = fit_rf.predict(test_set)

# Test Set Metrics
test_crosstb_comp = pd.crosstab(index = test_class_set,
                           columns = predictions_rf)

# More human readable
test_crosstb = test_crosstb_comp.rename(columns= {0: 'Benign', 1: 'Malignant'})
test_crosstb.index = ['Benign', 'Malignant']
test_crosstb.columns.name = 'n = 114'

# Accurary Score for test set predictions
accuracy_rf = fit_rf.score(test_set, test_class_set)

# Here we calculate the test error rate!
test_error_rate_rf = 1 - accuracy_rf

predictions_prob = fit_rf.predict_proba(test_set)[:, 1]

# ROC Curve stuff
fpr2, tpr2, _ = roc_curve(test_class_set,
	predictions_prob)

auc_rf = auc(fpr2, tpr2)

# Uncomment to save your model as a pickle object!
#sys.path.append('../../models/pickle_models/model_rf.pkl')
#path_rf = sys.path[-1]
#
#joblib.dump(fit_rf, path_rf)

if __name__=='__main__':
	# Print model parameters
	print(fit_rf)

	hf.variable_importance(importances_rf, indices_rf)

	print('CROSS VALIDATION')

	# Cross validation
	hf.cross_val_metrics(fit_rf,
                  training_set,
                  class_set,
                  print_results = True)

	print('TEST SET CALCULATIONS')


	print(test_crosstb)

	print("Here is our mean accuracy on the test set:\n {0: 0.3f}"\
		.format(accuracy_rf))

	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_rf))
else:
	def return_rf():
		'''
		Function to output values created in script
		'''
		return fpr2, tpr2, auc_rf, predictions_rf, test_error_rate_rf

	mean_cv_rf, std_error_rf = hf.cross_val_metrics(fit_rf,
		training_set, class_set,
		print_results = False)
