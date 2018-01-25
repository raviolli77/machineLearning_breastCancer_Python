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


# Hyperparamter optimization
#np.random.seed(42)
#start = time.time()
#
#param_dist = {'max_depth': [2, 3, 4],
              #'bootstrap': [True, False],
              #'max_features': ['auto', 'sqrt', 'log2', None],
              #'criterion': ['gini', 'entropy']}
#
#cv_rf = GridSearchCV(fit_rf, cv = 10,
                     #param_grid=param_dist,
                     #n_jobs = 3)
#
#cv_rf.fit(training_set, class_set)
#print('Best Parameters using grid search: \n',
      #cv_rf.best_params_,
      #'\n')
#end = time.time()
#print('Time taken in grid search: {0: .2f}\n'.format(end - start))
## Set best parameters given by grid search
fit_rf.set_params(criterion = 'gini',
                  max_features = 'log2',
                  max_depth = 3,
                  n_estimators=400,
                  oob_score=True)
#
## Estimating Number of Trees
#min_estimators = 15
#max_estimators = 1000
#
#error_rate = {}
#
#for i in range(min_estimators, max_estimators + 1):
    #fit_rf.set_params(n_estimators=i)
    #fit_rf.fit(training_set, class_set)
#
    #oob_error = 1 - fit_rf.oob_score_
    #error_rate[i] = oob_error
#
## Convert dictionary to a pandas series for easy plotting
#oob_series = pd.Series(error_rate)
#
#fig, ax = plt.subplots(figsize=(10, 10))
#
#ax.set_axis_bgcolor('#fafafa')
#
#oob_series.plot(kind='line',
                #color = 'red')
#plt.axhline(0.055,
            #color='#875FDB',
           #linestyle='--')
#plt.axhline(0.05,
            #color='#875FDB',
           #linestyle='--')
#plt.xlabel('n_estimators')
#plt.ylabel('OOB Error Rate')
#plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')
#plt.show()
#
#print('OOB Error rate for 400 trees is: {0:.5f} \n'\
#.format(oob_series[400]))
#
## Set final parameters
#fit_rf.set_params(n_estimators=400,
                  #bootstrap = True,
                  #warm_start=False,
                  #oob_score=False)

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

# ROC Curve stuff
fpr2, tpr2, _ = roc_curve(predictions_rf,
	test_class_set)

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

	hf.variable_importance_plot(importances_rf, indices_rf)

	print('HYPERPARAMETER OPTIMIZATION')

	print("Note: Remove commented code to see this section")
	print("chosen parameters: {'bootstrap': True, '4:45, criterion': 'entropy', \
	'max_depth': 4}\
	 \nElapsed time of optimization: 189.949 seconds")

		# start = time.time()

		# param_dist = {"max_depth": [2, 3, 4],
		# "bootstrap": [True, False],
		# "criterion": ["gini", "entropy"]}

		# gs_rf = GridSearchCV(fit_rf, cv = 10,
			# param_grid=param_dist)

		# gs_rf.fit(training_set, class_set)
		# print(gs_rf.best_params_)
		# end = time.time()
		# print(end - start)

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

	# ROC Curve
	hf.plot_roc_curve(fpr2, tpr2, auc_rf, 'rf')
	# Zoomed in ROC Curve
	hf.plot_roc_curve(fpr2, tpr2, auc_rf, 'rf',
		(-0.001, 0.2), (0.7, 1.05))
else:
	def return_rf():
		'''
		Function to output values created in script
		'''
		return fpr2, tpr2, auc_rf, predictions_rf, test_error_rate_rf

	mean_cv_rf, std_error_rf = hf.cross_val_metrics(fit_rf,
		training_set, class_set,
		print_results = False)
