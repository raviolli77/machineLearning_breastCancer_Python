#!/usr/bin/env python3

	#####################################################
	##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
	#####################################################

# Project by Raul Eulogio

# Project found at: https://www.inertia7.com/projects/3


"""
Model Evaluation
"""
# Import Packages -----------------------------------------------
import matplotlib.pyplot as plt
from knn import fit_knn
from random_forest import fit_rf
from neural_networks import fit_nn
from data_extraction import training_set, class_set
from data_extraction import test_set, test_class_set
from data_extraction import training_set_scaled, test_set_scaled
from helper_functions import cross_val_metrics
from produce_model_metrics import produce_model_metrics
from terminaltables import AsciiTable
from sklearn.metrics import classification_report



# Calling up metrics from the model scripts
# KNN -----------------------------------------------------------
metrics_knn = produce_model_metrics(fit_knn, test_set,
	test_class_set, 'knn')
# Call each value from dictionary
predictions_knn = metrics_knn['predictions']
accuracy_knn = metrics_knn['accuracy']
fpr = metrics_knn['fpr']
tpr = metrics_knn['tpr']
auc_knn = metrics_knn['auc']

# Test Error Rate
test_error_rate_knn = 1 - accuracy_knn

# Cross Validated Score
mean_cv_knn, std_error_knn = cross_val_metrics(fit_knn,
                                               training_set,
                                               class_set,
                                               'knn',
                                               print_results = False)

# RF ------------------------------------------------------------
metrics_rf = produce_model_metrics(fit_rf, test_set,
	test_class_set, 'rf')
# Call each value from dictionary
predictions_rf = metrics_rf['predictions']
accuracy_rf = metrics_rf['accuracy']
fpr2 = metrics_rf['fpr']
tpr2 = metrics_rf['tpr']
auc_rf = metrics_rf['auc']

# Test Error Rate
test_error_rate_rf = 1 - accuracy_rf

# Cross Validated Score
mean_cv_rf, std_error_rf = cross_val_metrics(fit_rf,
                                             training_set,
                                             class_set,
                                             'rf',
                                             print_results = False)

# NN ------------------------------------------------------------
metrics_nn = produce_model_metrics(fit_nn, test_set_scaled,
	test_class_set, 'nn')

# Call each value from dictionary
predictions_nn = metrics_nn['predictions']
accuracy_nn = metrics_nn['accuracy']
fpr3 = metrics_nn['fpr']
tpr3 = metrics_nn['tpr']
auc_nn = metrics_nn['auc']

# Test Error Rate
test_error_rate_nn = 1 - accuracy_nn

# Cross Validated Score
mean_cv_nn, std_error_nn = cross_val_metrics(fit_nn,
                                             training_set_scaled,
                                             class_set,
                                             'nn',
                                             print_results = False)

# Main ----------------------------------------------------------
if __name__ == '__main__':
	# Populate list for human readable table from terminal line
	table_data = [[ 'Model/Algorithm', 'Test Error Rate',
                'False Negative for Test Set', 'Area under the Curve for ROC',
                'Cross Validation Score'],
               ['Kth Nearest Neighbor',
                round(test_error_rate_knn, 3),
                5,
                round(auc_knn, 3),
                "Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
                .format(mean_cv_knn, std_error_knn)],
               [ 'Random Forest',
                round(test_error_rate_rf, 3),
                3,
                round(auc_rf, 3),
                "Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
				.format(mean_cv_rf, std_error_rf)],
               [ 'Neural Networks' ,
                round(test_error_rate_nn, 3),
                1,
                round(auc_nn, 3),
                "Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
				.format(mean_cv_nn, std_error_nn)]]

	# convert to AsciiTable from terminaltables package
	table = AsciiTable(table_data)

	target_names = ['Benign', 'Malignant']

	print('Classification Report for Kth Nearest Neighbor:')
	print(classification_report(predictions_knn,
		test_class_set,
		target_names = target_names))

	print('Classification Report for Random Forest:')
	print(classification_report(predictions_rf,
		test_class_set,
		target_names = target_names))

	print('Classification Report for Neural Networks:')
	print(classification_report(predictions_nn,
		test_class_set,
		target_names = target_names))

	print("Comparison of different logistics relating to model evaluation:")
	print(table.table)

	# Plotting ROC Curves----------------------------------------
	f, ax = plt.subplots(figsize=(10, 10))

	plt.plot(fpr, tpr, label='Kth Nearest Neighbor ROC Curve (area = {0: .3f})'\
		.format(auc_knn),
         	color = 'deeppink',
         	linewidth=1)
	plt.plot(fpr2, tpr2,label='Random Forest ROC Curve (area = {0: .3f})'\
		.format(auc_rf),
         	color = 'red',
         	linestyle=':',
         	linewidth=2)
	plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve (area = {0: .3f})'\
		.format(auc_nn),
         	color = 'purple',
         	linestyle=':',
         	linewidth=3)

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve Comparison For All Models')
	plt.legend(loc="lower right")
	plt.show()

	# Zoomed in
	f, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr, tpr, label='Kth Nearest Neighbor ROC Curve  (area = {0: .3f})'\
		.format(auc_knn),
         	color = 'deeppink',
         	linewidth=1)
	plt.plot(fpr2, tpr2,label='Random Forest ROC Curve  (area = {0: .3f})'\
		.format(auc_rf),
         	color = 'red',
         	linestyle=':',
         	linewidth=3)
	plt.plot(fpr3, tpr3,label='Neural Networks ROC Curve  (area = {0: .3f})'\
		.format(auc_nn),
         	color = 'purple',
         	linestyle=':',
         	linewidth=3)

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.001, 0.2])
	plt.ylim([0.7, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve Comparison For All Models (Zoomed)')
	plt.legend(loc="lower right")
	plt.show()

	print('fin \n:)')
