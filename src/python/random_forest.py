#!/usr/bin/env python3

#####################################################
##    WISCONSIN BREAST CANCER - MACHINE LEARNING   ##
#####################################################
#
# Project by Raul Eulogio
#
# Project found at: https://www.inertia7.com/projects/3
#

"""
Random Forest Classification
"""
# Import Packages -----------------------------------------------
import time
import sys
from numpy import argsort
import pandas as pd
import helper_functions as hf
from data_extraction import names_index
from data_extraction import training_set, class_set
from data_extraction import test_set, test_class_set
from sklearn.ensemble import RandomForestClassifier
from produce_model_metrics import produce_model_metrics

# Fitting Random Forest -----------------------------------------
# Set the random state for reproducibility
fit_rf = RandomForestClassifier(random_state=42)

## Set best parameters given by grid search
fit_rf.set_params(criterion = 'gini',
                  max_features = 'log2',
                  max_depth = 3,
                  n_estimators=400)

# Fit model on training data
fit_rf.fit(training_set,
            class_set)

# Tree Specific -------------------------------------------------

# Extracting feature importance
var_imp_rf = hf.variable_importance(fit_rf)

importances_rf = var_imp_rf['importance']

indices_rf = var_imp_rf['index']

if __name__=='__main__':
    # Print model parameters ------------------------------------
    print(fit_rf, '\n')

    # Initialize function for metrics ---------------------------
    fit_dict_rf = produce_model_metrics(fit_rf,
                                        test_set,
                                        test_class_set,
                                        'rf')

    # Extract each piece from dictionary
    predictions_rf = fit_dict_rf['predictions']
    accuracy_rf = fit_dict_rf['accuracy']
    auc_rf = fit_dict_rf['auc']

    print("Hyperparameter Optimization:")
    print("chosen parameters: \n \
    {'max_features': 'log2', \n \
    'max_depth': 3, \n \
    'bootstrap': True, \n \
    'criterion': 'gini'}")
    print("Note: Remove commented code to see this section \n")

	# np.random.seed(42)
	# start = time.time()
	# param_dist = {'max_depth': [2, 3, 4],
	#              'bootstrap': [True, False],
	#              'max_features': ['auto', 'sqrt',
    # 'log2', None],
	#              'criterion': ['gini', 'entropy']}
	# cv_rf = GridSearchCV(fit_rf, cv = 10,
    #	                 param_grid=param_dist,
	#                     n_jobs = 3)
	# cv_rf.fit(training_set, class_set)
	# print('Best Parameters using grid search: \n',
    #	cv_rf.best_params_)
	# end = time.time()
	# print('Time taken in grid search: {0: .2f}'\
    #.format(end - start))

    # Test Set Calculations -------------------------------------
    # Test error rate
    test_error_rate_rf = 1 - accuracy_rf

    # Confusion Matrix
    test_crosstb = hf.create_conf_mat(test_class_set,
        predictions_rf)

    # Print Variable Importance
    hf.print_var_importance(importances_rf, indices_rf, names_index)

    # Cross validation
    print('Cross Validation:')
    hf.cross_val_metrics(fit_rf,
                         training_set,
                         class_set,
                         'rf',
                         print_results = True)

    print('Confusion Matrix:')
    print(test_crosstb, '\n')

    print("Here is our mean accuracy on the test set:\n {0: 0.3f}"\
        .format(accuracy_rf))

    print("The test error rate for our model is:\n {0: .3f}"\
        .format(test_error_rate_rf))
    import pdb
    pdb.set_trace()
