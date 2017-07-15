#!/usr/bin/env python3

	#####################################################
	##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
	#####################################################

# Project by Raul Eulogio

# Project found at: https://www.inertia7.com/projects/3


"""
Neural Networks Classification
"""

import time
import sys, os
from helper_functions import *
from sklearn.neural_network import MLPClassifier # Neural Networks
from sklearn.model_selection import KFold, cross_val_score # Cross validation
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # Calculating Area Under Curve for ROC's!


fit_NN = MLPClassifier(solver='lbfgs', 
	hidden_layer_sizes = (12, ),
	activation='tanh',
	learning_rate_init=0.05, 
	random_state=42)

fit_NN.fit(training_set_scaled, 
	class_set_scaled['diagnosis'])	

predictions_NN = fit_NN.predict(test_set_scaled)

accuracy_NN = fit_NN.score(test_set_scaled, 
	test_class_set_scaled['diagnosis'])

# Here we calculate the test error rate!
test_error_rate_NN = 1 - accuracy_NN

# ROC Curve stuff
fpr3, tpr3, _ = roc_curve(predictions_NN, test_class_set_scaled)

auc_nn = auc(fpr3, tpr3)

if __name__ == '__main__':	
	print('''
	##################################
	##         FITTING MLP          ##
	##################################
	'''
	)	
	print(fit_NN)	
	print('''
	############################################
	##       HYPERPARAMETER OPTIMIZATION      ##
	############################################
	'''
	)	
	print("Note: Remove commented code to see this section")
	print("chosen parameters: \n \
	{'hidden_layer_sizes': 12, \n \
	'activation': 'tanh', \n \
	'learning_rate_init': 0.05} \
		\nEstimated time: 31.019 seconds")	
	# start = time.time()
	# gs = GridSearchCV(fit_NN, cv = 10,
		# param_grid={
		# 'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
		# 'hidden_layer_sizes': [4, 8, 12],
		# 'activation': ["relu", "identity", "tanh", "logistic"]})	 	
	# gs.fit(training_set_scaled, class_set_scaled['diagnosis'])
	# print(gs.best_params_)
	# end = time.time()
	# print(end - start)	
	print('''
	################################
	##      CROSS VALIDATION      ##
	################################
	'''
	)	

	test_thing = crossVD(fit_NN, training_set_scaled, 
		class_set_scaled['diagnosis'], 
		print_results = True)	

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)	
	print(pd.crosstab(predictions_NN, 
		test_class_set_scaled['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))

	print("Here is our mean accuracy on the test set:\n {0: .3f}"\
		.format(accuracy_NN))			
	
	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_NN))	
	
	# ROC Curve
	plotROC(fpr3, tpr3, auc_nn, 2)	
	
	# Zoomed in ROC Curve
	plotROCZoom(fpr3, tpr3, auc_nn, 2)
else:
	def return_nn():
		return fpr3, tpr3, auc_nn, predictions_NN, test_error_rate_NN
	
	# Keep Cross validation metrics 
	mean_cv_nn, std_cv_nn = crossVD(fit_NN, 
		training_set_scaled, 
		class_set_scaled['diagnosis'], 
		print_results = False)