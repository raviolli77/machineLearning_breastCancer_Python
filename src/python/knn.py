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
Kth Nearest Neighbor Classification
"""
# Import Packages -----------------------------------------------
import sys, os
import pandas as pd
import helper_functions as hf
from data_extraction import training_set, class_set
from data_extraction import test_set, test_class_set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from produce_model_metrics import produce_model_metrics

# Fitting model
fit_knn = KNeighborsClassifier(n_neighbors=3)

# Training model
fit_knn.fit(training_set,
	class_set)
# ---------------------------------------------------------------
if __name__ == '__main__':
	# Print model parameters ------------------------------------
	print(fit_knn, '\n')

	# Optimal K -------------------------------------------------
	# Inspired by:
	# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

	myKs = []
	for i in range(0, 50):
		if (i % 2 != 0):
			myKs.append(i)

	cross_vals = []
	for k in myKs:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn,
                           training_set,
                           class_set,
                           cv = 10,
                           scoring='accuracy')
		cross_vals.append(scores.mean())

	MSE = [1 - x for x in cross_vals]
	optimal_k = myKs[MSE.index(min(MSE))]
	print("Optimal K is {0}".format(optimal_k), '\n')

	# Initialize function for metrics ---------------------------
	fit_dict_knn = produce_model_metrics(fit_knn,
                                      test_set,
                                      test_class_set,
                                      'knn')
	# Extract each piece from dictionary
	predictions_knn = fit_dict_knn['predictions']
	accuracy_knn = fit_dict_knn['accuracy']
	auc_knn = fit_dict_knn['auc']

	# Test Set Calculations -------------------------------------
	# Test error rate
	test_error_rate_knn = 1 - accuracy_knn

	# Confusion Matrix
	test_crosstb = hf.create_conf_mat(test_class_set,
		predictions_knn)

	print('Cross Validation:')
	hf.cross_val_metrics(fit_knn,
                      training_set,
                      class_set,
                      'knn', 
                      print_results = True)

	print('Confusion Matrix:')
	print(test_crosstb, '\n')

	print("Here is our accuracy for our test set:\n {0: .3f}"\
	.format(accuracy_knn))

	print("The test error rate for our model is:\n {0: .3f}"\
	.format(test_error_rate_knn))
