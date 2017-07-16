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

import time
import sys, os
import pandas as pd
import helper_functions as hf
from helper_functions import training_set_scaled, class_set_scaled
from helper_functions import test_set_scaled, test_class_set_scaled
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc 
from sklearn.externals import joblib

# Fit model 
fit_nn = MLPClassifier(solver='lbfgs', 
	hidden_layer_sizes = (12, ),
	activation='tanh',
	learning_rate_init=0.05, 
	random_state=42)

# Train model on training set
fit_nn.fit(training_set_scaled, 
	class_set_scaled['diagnosis'])	

predictions_nn = fit_nn.predict(test_set_scaled)

accuracy_nn = fit_nn.score(test_set_scaled, 
	test_class_set_scaled['diagnosis'])

# Here we calculate the test error rate!
test_error_rate_nn = 1 - accuracy_nn

# ROC Curve stuff
fpr3, tpr3, _ = roc_curve(predictions_nn, test_class_set_scaled)

auc_nn = auc(fpr3, tpr3)

# Uncomment to save your model as a pickle object!
# joblib.dump(fit_nn, 'pickle_models/model_nn.pkl')

if __name__ == '__main__':	
	print('''
	##################################
	##         FITTING MLP          ##
	##################################
	'''
	)	
	print(fit_nn)	
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
	# gs = GridSearchCV(fit_nn, cv = 10,
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

	test_thing = hf.cross_val_metrics(fit_nn, training_set_scaled, 
		class_set_scaled['diagnosis'], 
		print_results = True)	

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)	
	print(pd.crosstab(predictions_nn, 
		test_class_set_scaled['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))

	print("Here is our mean accuracy on the test set:\n {0: .3f}"\
		.format(accuracy_nn))			
	
	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_nn))	
	
	# ROC Curve
	hf.plot_roc_curve(fpr3, tpr3, auc_nn, 'nn')	
	
	# Zoomed in ROC Curve
	hf.plot_roc_curve(fpr3, tpr3, auc_nn, 'nn',  
		(-0.001, 0.2), (0.7, 1.05))
else:
	def return_nn():
		'''
		Function to output values created in script 
		'''
		return fpr3, tpr3, auc_nn, predictions_nn, test_error_rate_nn
	
	# Keep Cross validation metrics 
	mean_cv_nn, std_error_nn = hf.cross_val_metrics(fit_nn, 
		training_set_scaled, 
		class_set_scaled['diagnosis'], 
		print_results = False)