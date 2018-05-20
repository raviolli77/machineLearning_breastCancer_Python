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
Neural Networks Classification
"""
# Import Packages -----------------------------------------------
import sys, os
import pandas as pd
import helper_functions as hf
from data_extraction import training_set_scaled, class_set
from data_extraction import test_set_scaled, test_class_set
from sklearn.neural_network import MLPClassifier
from produce_model_metrics import produce_model_metrics

# Fitting Neural Network ----------------------------------------
# Fit model
fit_nn = MLPClassifier(solver='lbfgs',
	hidden_layer_sizes = (12, ),
	activation='tanh',
	learning_rate_init=0.05,
	random_state=42)

# Train model on training set
fit_nn.fit(training_set_scaled,
	class_set)

if __name__ == '__main__':
	# Print model parameters ------------------------------------
	print(fit_nn, '\n')

	# Initialize function for metrics ---------------------------
	fit_dict_nn = produce_model_metrics(fit_nn, test_set_scaled,
	test_class_set, 'nn')
	# Extract each piece from dictionary
	predictions_nn = fit_dict_nn['predictions']
	accuracy_nn = fit_dict_nn['accuracy']
	auc_nn = fit_dict_nn['auc']


	print("Hyperparameter Optimization:")
	print("chosen parameters: \n \
	{'hidden_layer_sizes': 12, \n \
	'activation': 'tanh', \n \
	'learning_rate_init': 0.05}")
	print("Note: Remove commented code to see this section \n")

	# from sklearn.model_selection import GridSearchCV
	# import time
	# start = time.time()
	# gs = GridSearchCV(fit_nn, cv = 10,
		# param_grid={
		# 'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
		# 'hidden_layer_sizes': [4, 8, 12],
		# 'activation': ["relu", "identity", "tanh", "logistic"]})
	# gs.fit(training_set_scaled, class_set)
	# print(gs.best_params_)
	# end = time.time()
	# print(end - start)

	# Test Set Calculations -------------------------------------
	# Test error rate
	test_error_rate_nn = 1 - accuracy_nn

	# Confusion Matrix
	test_crosstb = hf.create_conf_mat(test_class_set,
		predictions_nn)

	# Cross validation
	print("Cross Validation:")

	hf.cross_val_metrics(fit_nn,
                      training_set_scaled,
                      class_set,
                      'nn',
                      print_results = True)

	print('Confusion Matrix:')
	print(test_crosstb, '\n')

	print("Here is our mean accuracy on the test set:\n {0: .3f}"\
		.format(accuracy_nn))

	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_nn))
