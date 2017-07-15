#!/usr/bin/env python3

	#####################################################
	##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
	#####################################################

# Project by Raul Eulogio

# Project found at: https://www.inertia7.com/projects/3


"""
Helper Functions Script
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Create training and test sets
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from sklearn.model_selection import KFold, cross_val_score # Cross validation
from urllib.request import urlopen # Get data from UCI Machine Learning Repository


UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases\
/breast-cancer-wisconsin/wdbc.data'

names = ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
         'radius_se', 'texture_se', 'perimeter_se', 
         'area_se', 'smoothness_se', 'compactness_se', 
         'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 

breastCancer = pd.read_csv(urlopen(UCI_data_URL), names=names)

# Setting 'id_number' as our index
breastCancer.set_index(['id_number'], inplace = True) 

# Converted to binary to help later on with models and plots
breastCancer['diagnosis'] = breastCancer['diagnosis'].map({'M':1, 'B':0})

# For later use in CART models
namesInd = names[2:]

	###################################
	##    HELPER FUNCTIONS SCRIPT    ##
	###################################
def classImbalance(dataFrame, item):
    '''
    Goal of this function:
    Loops through the Dx to print percentage of class distributions 
    w.r.t. the length of the data set
    '''
    i = 0
    n = 0
    perMal = 0 
    perBeg = 0
    for item in dataFrame[item]:
        if (item == 1):
            i += 1
        elif (item == 0):
            n += 1
    perMal = (i/len(dataFrame)) * 100
    perBeg = (n/len(dataFrame)) * 100
    print("The percentage of Malignant Dx is: {0:.3f}%".format(perMal)) 
    print("The percentage of Begnin Dx is: {0:.3f}%".format(perBeg))


def pltBoxPlot(minxLim, maxXLim, dataFrame, dataSet):
	f, ax = plt.subplots(figsize=(11, 15))
	
	ax.set_axis_bgcolor('#fafafa')
	ax.set(xlim=(minxLim, maxXLim))
	plt.ylabel('Dependent Variables')
	plt.title("Box Plot of {0} Data Set".format(dataSet))
	ax = sns.boxplot(data = dataFrame, orient = 'h', palette = 'Set2')
	
	plt.show()
	plt.close()



def normalize_df(frame):
	'''
	Helper function to Normalize data set
	Intializes an empty data frame which will normalize all floats types
	and just append the non-float types so basically the class in our data frame
	'''
	breastCancerNorm = pd.DataFrame()
	for item in frame:
		if item in frame.select_dtypes(include=[np.float]):
			breastCancerNorm[item] = ((frame[item] - frame[item].min()) / 
			(frame[item].max() - frame[item].min()))
		else: 
			breastCancerNorm[item] = frame[item]
	return breastCancerNorm

	############################################
	##    CREATING TRAINING AND TEST SETS     ##
	############################################

def splitSets(breastCancer):
	'''
	Creating a helper function to split the data into a 80-20
	training and test split. As well as creating class set and 
	class test set, which is what I call them since Python's
	naming for these gets confusing
	'''
	breastCancerNorm = normalize_df(breastCancer)
	
	# Here we do a 80-20 split for our training and test set
	train, test = train_test_split(breastCancerNorm, 
                               	test_size = 0.20, 
                               	random_state = 42)
	
	# Create the training test omitting the diagnosis
	training_set = train.ix[:, train.columns != 'diagnosis']
	# Next we create the class set (Called target in Python Documentation)
	# Note: This was confusing af to figure out cus the documentation is low-key kind of shitty
	class_set = train.ix[:, train.columns == 'diagnosis']
	
	# Next we create the test set doing the same process as the training set
	test_set = test.ix[:, test.columns != 'diagnosis']
	test_class_set = test.ix[:, test.columns == 'diagnosis']
	return training_set, class_set, test_set, test_class_set

# CREATE APPROPRIATE SETS FOR MODELING
training_set, class_set, test_set, test_class_set = splitSets(breastCancer)

# Scaling dataframe
breastCancerNorm = normalize_df(breastCancer)
training_set_scaled, class_set_scaled, test_set_scaled, \
test_class_set_scaled = splitSets(breastCancerNorm)

def varImport(names, importance, indices):
	'''
	Helper function that returns the variable importance for CART models
	'''
	print("Feature ranking:")
	
	for f in range(30):
		i = f
		print("%d. The feature '%s' has a Information Gain of %f" % (f + 1, 
																	names[indices[i]], 
																	importance[indices[f]]))



def varImportPlot(index, feature_space, indRf):
	# PLOTTING VARIABLE IMPORTANCE
	f, ax = plt.subplots(figsize=(11, 11))
	
	ax.set_axis_bgcolor('#fafafa')
	plt.title('Feature importances for Random Forest Model')
	plt.barh(index, indRf,
		align="center", 
		color = '#875FDB')
	plt.yticks(index, 
		feature_space)
	
	plt.ylim(-1, 30)
	plt.xlim(0, 0.15)
	plt.xlabel('Information Gain Entropy')
	plt.ylabel('Feature')
	
	plt.show()
	plt.close()

	############################################
	##        ROC Curve Helper Function       ##      
	############################################

def plotROC(fpr, tpr, auc, i):
	'''
	Generates the respective ROC Curve for each model
	Where the legend is as such in dictionary form:
	{0: 'KNN', 1: 'Decision Tree', 2: 'Random Forest', 3: 'Neural Networks'}
	'''
	colors = ['deeppink', 'red', 'purple']
	method = ['Kth Nearest Neighbor', 'Random Forest', 
	'Neural Network']

	# ROC Curve
	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr, tpr, 
		color = colors[i], 
		linewidth=1)	

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'\
		.format(method[i], auc))		
	plt.show()
	plt.close()

	######################################################
	##        Zoomed in ROC Curve Helper Function       ##      
	######################################################

def plotROCZoom(fpr, tpr, auc, i):	
	'''
	Generates the respective ROC Curve for each model Zoomed in!
	Where the legend is as such in dictionary form:
	{0: 'KNN', 1: 'Decision Trees', 2: 'Random Forest', 3: 'Neural Networks'}
	'''
	colors = ['deeppink', 'red', 'purple']
	method = ['Kth Nearest Neighbor', 'Random Forest', 
	'Neural Network']

	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr, tpr, 
		color = colors[i], 
		linewidth=1)

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.001, 0.2])
	plt.ylim([0.7, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Zoomed in ROC Curve For {0} (AUC = {1: 0.3f})'\
		.format(method[i], auc))		
	plt.show()
	plt.close()


def crossVD(fit, test_set, test_class_set, print_results = True):
	'''
	Helper function helps automate cross validation processes
	'''
	n = KFold(n_splits=10)
	scores = cross_val_score(fit, 
                         test_set, 
                         test_class_set, 
                         cv = n)
	if print_results:
		print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
			.format(scores.mean(), scores.std() / 2))
	else:
		return scores.mean(), scores.std() / 2