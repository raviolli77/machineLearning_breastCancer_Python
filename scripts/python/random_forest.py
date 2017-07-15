#!/usr/bin/env python3

	#####################################################
	##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
	#####################################################

# Project by Raul Eulogio

# Project found at: https://www.inertia7.com/projects/3


"""
Random Forest Classification
"""

import time
import sys, os
from helper_functions import *
from sklearn.model_selection import KFold, cross_val_score # Cross validation
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # Calculating Area Under Curve for ROC's!


# Fitting Random Forest
fit_RF = RandomForestClassifier(random_state = 42, 
	bootstrap=True,
	max_depth=4,
	criterion='entropy',
	n_estimators = 500)

fit_RF.fit(training_set, 
	class_set['diagnosis'])

importancesRF = fit_RF.feature_importances_

# Create indices for importance of features
indicesRF = np.argsort(importancesRF)[::-1]

# Sort by Decreasing order
indRf = sorted(importancesRF) 

index = np.arange(30)	

predictions_RF = fit_RF.predict(test_set)	
accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])

# Here we calculate the test error rate!
test_error_rate_RF = 1 - accuracy_RF

# ROC Curve stuff
fpr2, tpr2, _ = roc_curve(predictions_RF, 
	test_class_set)	

auc_rf = auc(fpr2, tpr2)

# Fitting model 
if __name__=='__main__':
	print(fit_RF)

	varImport(namesInd, importancesRF, indicesRF)
	
	feature_space = []
	for i in range(29, -1, -1):
		feature_space.append(namesInd[indicesRF[i]])
	
	varImportPlot(index, feature_space, indRf)
	
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
	
	# CROSS VALIDATION
	crossVD(fit_RF, training_set, class_set['diagnosis'], 
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
	plotROC(fpr2, tpr2, auc_rf, 1)
		# Zoomed in ROC Curve
	plotROCZoom(fpr2, tpr2, auc_rf, 1)
else:
	def return_rf():
		return fpr2, tpr2, auc_rf, predictions_RF, test_error_rate_RF

	mean_cv_rf, std_cv_rf = crossVD(fit_RF, 
		training_set, 
		class_set['diagnosis'], 
		print_results = False)