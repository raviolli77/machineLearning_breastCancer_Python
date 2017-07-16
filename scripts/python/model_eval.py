#!/usr/bin/env python3

	#####################################################
	##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
	#####################################################

# Project by Raul Eulogio

# Project found at: https://www.inertia7.com/projects/3


"""
Model Evaluation 
"""

from random_forest import *
from knn import *
from neural_networks import *
from terminaltables import AsciiTable
from sklearn.metrics import classification_report

# Calling up metrics from the model scripts
fpr, tpr, auc_knn, predictions, test_error_rate = return_knn()

fpr2, tpr2, auc_rf, predictions_rf, test_error_rate_rf = return_rf()

fpr3, tpr3, auc_nn, predictions_nn, test_error_rate_nn = return_nn()

if __name__ == '__main__':
	# Populate list for human readable table from terminal line
	table_data = [[ 'Model/Algorithm', 'Test Error Rate', 
		'False Negative for Test Set', 'Area under the Curve for ROC', 
		'Cross Validation Score'],
		['Kth Nearest Neighbor',  round(test_error_rate, 3), 5, 
		round(auc_knn, 3), "Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
				.format(mean_cv_knn, std_error_knn)],
		[ 'Random Forest', round(test_error_rate_rf, 3), 3, 
		round(auc_rf, 3), "Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
				.format(mean_cv_rf, std_error_rf)], 
		[ 'Neural Networks' ,  round(test_error_rate_nn, 3),  1, 
		round(auc_nn, 3), "Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
				.format(mean_cv_nn, std_error_nn)]]
	
	# convert to AsciiTable from terminaltables package
	table = AsciiTable(table_data)	
	
	target_names = ['Benign', 'Malignant']
	
	print('Classification Report for Kth Nearest Neighbor:')
	print(classification_report(predictions, 
		test_class_set['diagnosis'], 
		target_names = target_names))
	
	print('Classification Report for Random Forest:')
	print(classification_report(predictions_rf, 
		test_class_set['diagnosis'], 
		target_names = target_names))
	
	print('Classification Report for Neural Networks:')
	print(classification_report(predictions_nn, 
		test_class_set_scaled['diagnosis'], 
		target_names = target_names))
	
	print("Comparison of different logistics relating to model evaluation:")
	print(table.table)
	
	# Plotting ROC Curves
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