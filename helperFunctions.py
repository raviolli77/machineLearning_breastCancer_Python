#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # Create training and test sets
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from sklearn.model_selection import KFold, cross_val_score # Cross validation

	###################################
	##    HELPER FUNCTIONS SCRIPT    ##
	###################################

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
	plt.xlim(0, 0.13)
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


def crossVD(fit, test_set, test_class_set):
	'''
	Helper function helps automate cross validation processes
	'''
	n = KFold(n_splits=10)
	scores = cross_val_score(fit, 
                         test_set, 
                         test_class_set, 
                         cv = n)

	print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
		.format(scores.mean(), scores.std() / 2))

