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
from helper_functions import training_set, class_set, test_set, test_class_set
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc 
from sklearn.externals import joblib

# Fitting Random Forest
fit_RF = RandomForestClassifier(random_state = 42, 
	bootstrap=True,
	max_depth=4,
	criterion='entropy',
	n_estimators = 500)

# Training Model
fit_RF.fit(training_set, 
	class_set['diagnosis'])

# Extracting feature importance
import_rf = fit_RF.feature_importances_

# Create indices for importance of features
ind_rf = np.argsort(import_rf)[::-1]

# Sort by Decreasing order
import_rf_desc = sorted(import_rf) 	

# Predictions for test set
predictions_RF = fit_RF.predict(test_set)	
accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])

# Here we calculate the test error rate!
test_error_rate_RF = 1 - accuracy_RF

# ROC Curve stuff
fpr2, tpr2, _ = roc_curve(predictions_RF, 
	test_class_set)	

auc_rf = auc(fpr2, tpr2)

# Uncomment to save your model as a pickle object!
# joblib.dump(fit_RF, 'pickle_models/model_rf.pkl')

if __name__=='__main__':
	# Print model parameters
	print(fit_RF)

	hf.variable_importance(import_rf, ind_rf)
	
	hf.variable_importance_plot(import_rf_desc, ind_rf)

	print('''
	############################################
	##      HYPERPARAMETER OPTIMIZATION       ##
	############################################
	'''
	)
	
	print("Note: Remove commented code to see this section")
	print("chosen parameters: {'bootstrap': True, 'criterion': 'entropy', \
	'max_depth': 4}\
	 	\nElapsed time of optimization: 189.949 seconds")
	
		# start = time.time()
	
		# param_dist = {"max_depth": [2, 3, 4],
		# "bootstrap": [True, False],
		# "criterion": ["gini", "entropy"]}
	
		# gs_rf = GridSearchCV(fit_RF, cv = 10,
			# param_grid=param_dist)
	
		# gs_rf.fit(training_set, class_set['diagnosis'])
		# print(gs_rf.best_params_)
		# end = time.time()
		# print(end - start)
	
	print('''
		###############################
		##      CROSS VALIDATION     ##
		###############################
		'''
		)
	
	# Cross validation 
	hf.cross_val_metrics(fit_RF, training_set, class_set['diagnosis'], 
		print_results = True)
	
	print('''
		###############################
		##   TEST SET CALCULATIONS   ##
		###############################
		'''
		)
		
	print(pd.crosstab(predictions_RF, 
			test_class_set['diagnosis'], 
			rownames=['Predicted Values'], 
			colnames=['Actual Values']))
	
	print("Here is our mean accuracy on the test set:\n {0: 0.3f}"\
		.format(accuracy_RF))
	
	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_RF))
		
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
		return fpr2, tpr2, auc_rf, predictions_RF, test_error_rate_RF

	mean_cv_rf, std_error_rf = hf.cross_val_metrics(fit_RF, 
		training_set, 
		class_set['diagnosis'], 
		print_results = False)