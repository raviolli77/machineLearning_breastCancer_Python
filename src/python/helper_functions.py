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
Helper Functions Script
"""
# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from urllib.request import urlopen

# Loading data and cleaning dataset
UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean',
         'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean',
         'concavity_mean','concave_points_mean',
         'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se',
         'area_se', 'smoothness_se', 'compactness_se',
         'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se',
         'radius_worst', 'texture_worst',
         'perimeter_worst', 'area_worst',
         'smoothness_worst', 'compactness_worst',
         'concavity_worst', 'concave_points_worst',
         'symmetry_worst', 'fractal_dimension_worst']

dx = ['Malignant', 'Benign']

breast_cancer = pd.read_csv(urlopen(UCI_data_URL), names=names)

# Setting 'id_number' as our index
breast_cancer.set_index(['id_number'], inplace = True)

# Converted to binary to help later on with models and plots
breast_cancer['diagnosis'] = breast_cancer['diagnosis'].map({'M':1, 'B':0})

for col in breast_cancer:
	pd.to_numeric(col, errors='coerce')

# For later use in CART models
names_index = names[2:]

def print_dx_perc(data_frame, col):
    """Function used to print class distribution for our data set"""
    dx_vals = data_frame[col].value_counts()
    dx_vals = dx_vals.reset_index()
    # Create a function to output the percentage
    f = lambda x, y: 100 * (x / sum(y))
    for i in range(0, len(dx)):
        print('{0} accounts for {1:.2f}% of the diagnosis class'\
              .format(dx[i], f(dx_vals[col].iloc[i],
                               dx_vals[col])))

def plot_box_plot(data_frame, data_set, xlim=None):
	"""
	Purpose
	----------
	Creates a seaborn boxplot including all dependent
	variables and includes x limit parameters

	Parameters
	----------
	* data_frame :	Name of pandas.dataframe
	* data_set :		Name of title for the boxplot
	* xlim : 	Set upper and lower x-limits
	"""
	f, ax = plt.subplots(figsize=(11, 15))

	ax.set_axis_bgcolor('#fafafa')
	if xlim is not None:
		plt.xlim(*xlim)
	plt.ylabel('Dependent Variables')
	plt.title("Box Plot of {0} Data Set"\
		.format(data_set))
	ax = sns.boxplot(data = data_frame,
		orient = 'h',
		palette = 'Set2')

	plt.show()
	plt.close()

def normalize_data_frame(data_frame):
	"""
	Purpose
	----------
	Function created to normalize data set.
	Intializes an empty data frame which will normalize all floats types
	and append the non-float types. Application is very specific
	to this dataset, can be changed to include integer types in the
	normalization.

	Parameters
	----------
	* data_frame: 	Name of pandas.dataframe


	Returns
	----------
	* data_frame_norm:	Normalized dataframe values ranging (0, 1)
	"""
	data_frame_norm = pd.DataFrame()
	for col in data_frame:
		if col in data_frame.select_dtypes(include=[np.float]):
			data_frame_norm[col]=((data_frame[col] - data_frame[col].min()) /
			(data_frame[col].max() - data_frame[col].min()))
		else:
			data_frame_norm[col] = data_frame[col]
	return data_frame_norm


feature_space = breast_cancer.iloc[:, breast_cancer.columns != 'diagnosis']
feature_class = breast_cancer.iloc[:, breast_cancer.columns == 'diagnosis']


training_set, test_set, class_set, test_class_set = train_test_split(feature_space,
                                                                    feature_class,
                                                                    test_size = 0.20,
                                                                    random_state = 42)

# Cleaning test sets to avoid future warning messages
class_set = class_set.values.ravel()
test_class_set = test_class_set.values.ravel()

# Scaling dataframe
scaler = MinMaxScaler()

scaler.fit(training_set)

training_set_scaled = scaler.fit_transform(training_set)
test_set_scaled = scaler.transform(test_set)

def variable_importance(importance, indices):
	"""
	Purpose
	----------
	Prints dependent variable names ordered from largest to smallest
	based on information gain for CART model.

	Parameters
	----------
	* names: 	Name of columns included in model
	* importance: 	Array returned from feature_importances_ for CART
					models organized by dataframe index
	* indices: 	Organized index of dataframe from largest to smallest
				based on feature_importances_
	"""
	print("Feature ranking:")

	for f in range(30):
		i = f
		print("%d. The feature '%s' has a Mean Decrease in Gini of %f" % (f + 1,
			names_index[indices[i]],
			importance[indices[f]]))

def variable_importance_plot(importance, indices):
    """
    Purpose
    ----------
    Prints bar chart detailing variable importance for CART model
    NOTE: feature_space list was created because the bar chart
    was transposed and index would be in incorrect order.

    Parameters
    ----------
    importance_desc: Array returned from feature_importances_ for CART
                    models organized in descending order

    indices: Organized index of dataframe from largest to smallest
                    based on feature_importances_
    Returns:
    ----------
    Returns variable importance plot in descending order
    """
    index = np.arange(len(names_index))

    importance_desc = sorted(importance)
    feature_space = []
    for i in range(29, -1, -1):
        feature_space.append(names_index[indices[i]])

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_axis_bgcolor('#fafafa')
    plt.title('Feature importances for Random Forest Model\
    \nBreast Cancer (Diagnostic)')
    plt.barh(index,
         importance_desc,
         align="center",
         color = '#875FDB')
    plt.yticks(index,
           feature_space)

    plt.ylim(-1, 30)
    plt.xlim(0, max(importance_desc) + 0.01)
    plt.xlabel('Mean Decrease in Gini')
    plt.ylabel('Feature')

    plt.show()
    plt.close()

def plot_roc_curve(fpr, tpr, auc, mod, xlim=None, ylim=None):
	'''
	Purpose
	----------
	Function creates ROC Curve for respective model given selected parameters.
	Optional x and y limits to zoom into graph

	Parameters
	----------
	* fpr: 	Array returned from sklearn.metrics.roc_curve for increasing
			false positive rates
	* tpr: 	Array returned from sklearn.metrics.roc_curve for increasing
			true positive rates
	* auc:	Float returned from sklearn.metrics.auc (Area under Curve)
	* mod: 	String represenation of appropriate model, can only contain the
			following: ['knn', 'rf', 'nn']
	* xlim:		Set upper and lower x-limits
	* ylim:		Set upper and lower y-limits
	'''
	mod_list = ['knn', 'rf', 'nn']
	method = [('Kth Nearest Neighbor', 'deeppink'), ('Random Forest', 'red'),
	('Neural Network', 'purple')]

	plot_title = ''
	color_value = ''
	for i in range(0, 3):
		if mod_list[i] == mod:
			plot_title = method[i][0]
			color_value = method[i][1]

	fig, ax = plt.subplots(figsize=(10, 10))
	ax.set_axis_bgcolor('#fafafa')

	plt.plot(fpr, tpr,
		color=color_value,
		linewidth=1)
	plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'\
		.format(plot_title, auc))

	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	if xlim is not None:
		plt.xlim(*xlim)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.show()
	plt.close()

def cross_val_metrics(fit, training_set, class_set, print_results = True):
	"""
	Purpose
	----------
	Function helps automate cross validation processes while including
	option to print metrics or store in variable

	Parameters
	----------
	* fit:	Fitted model
	* training_set: 	Dataframe containing 80% of original dataframe
	* class_set: 	Dataframe containing the respective target vaues
					for the training_set
	* print_results:	If true prints the metrics, else saves metrics as
	variables

	Returns
	----------
	* scores.mean(): 	Float representing cross validation score
	* scores.std() / 2: 	Float representing the standard error (derived
				from cross validation score's standard deviation)
	"""
	n = KFold(n_splits=10)
	scores = cross_val_score(fit,
                         training_set,
                         class_set,
                         cv = n)
	if print_results:
		print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
			.format(scores.mean(), scores.std() / 2))
	else:
		return scores.mean(), scores.std() / 2
