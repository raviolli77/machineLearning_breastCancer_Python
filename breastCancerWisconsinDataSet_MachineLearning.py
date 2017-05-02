#!/usr/bin/env python3

	#####################################################
	##    WISCONSIN BREAST CANCER MACHINE LEARNING     ##
	#####################################################

# Project by Raul Eulogio
import time
import sys, os
import numpy as np
import pandas as pd # Data frames
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from helperFunctions import *
from sklearn.model_selection import KFold, cross_val_score # Cross validation
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.neural_network import MLPClassifier # Neural Networks
from sklearn.model_selection import KFold, GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # Calculating Area Under Curve for ROC's!
from urllib.request import urlopen # Get data from UCI Machine Learning Repository

pd.set_option('display.max_columns', 500) # Included to show all the columns 
# since it is a fairly large data set
plt.style.use('ggplot') # Using ggplot2 style visuals 
# because that's how I learned my visuals 
# and I'm sticking to it!

	#################################
	##        LOADING DATA         ##
	#################################

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

breastCancer.set_index(['id_number'], inplace = True) # Setting 'id_number' as our index

# Converted to binary to help later on with models and plots
breastCancer['diagnosis'] = breastCancer['diagnosis'].map({'M':1, 'B':0})

# For later use in CART models
namesInd = names[2:]

# FUNCTION NOT WORKING WHEN PUT INSIDE 'helperfunctions.py'
# NEED TO LEARN HOW TO FIX

def classImbalance(item):
    '''
    Goal of this function:
    Loops through the Dx to print percentage of class distributions 
    w.r.t. the length of the data set
    '''
    i = 0
    n = 0
    perMal = 0 
    perBeg = 0
    for item in breastCancer[item]:
        if (item == 1):
            i += 1
        elif (item == 0):
            n += 1
    perMal = (i/len(breastCancer)) * 100
    perBeg = (n/len(breastCancer)) * 100
    print("The percentage of Malignant Dx is: {0:.3f}%".format(perMal)) 
    print("The percentage of Begnin Dx is: {0:.3f}%".format(perBeg))

	#################################
	##    EXPLORATORY ANALYSIS     ##
	#################################

def exploratoryAnalysis():
	'''
	Function shows various statistical calculations done 
	as a preliminary exploratory analysis 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py EA 
	'''
	print('''
	#########################################
	##    NUMERIC EXPLORATORY ANALYSIS     ##
	#########################################
	''')
	
	print('''
	########################################
	##    DATA FRAME SHAPE AND DTYPES     ##
	########################################
	''')

	print("Here's the dimensions of our data frame:\n", 
		breastCancer.shape)
	print("Here's the data types of our columns:\n", 
		breastCancer.dtypes)
	
	print("Some more statistics for our data frame: \n", 
		breastCancer.describe())

	print('''
	##########################################
	##      STATISTICS RELATING TO DX       ##
	##########################################
	''')
	# Let's look at the count of the new representations of our Dx's
	print("Count of the Dx:\n", breastCancer['diagnosis']\
		.value_counts())

	# Next let's use the helper function to show distribution of our data frame
	classImbalance('diagnosis')

def visualExplorAnalysis():
	'''
	Function shows various visual exploratory analysis plots
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py VEA 
	'''

	# Scatterplot Matrix
	# Variables chosen from Random Forest modeling.
	
	
	cols = ['concave_points_worst', 'concavity_mean', 
		'perimeter_worst', 'radius_worst', 
		'area_worst', 'diagnosis']

	sns.pairplot(breastCancer,
		x_vars = cols,
		y_vars = cols,
		hue = 'diagnosis', 
		palette = ('Red', '#875FDB'), 
		markers=["o", "D"])
	plt.title('Scatterplot Matrix')

	plt.show()
	plt.close()

	# Pearson Correlation Matrix
	corr = breastCancer.corr(method = 'pearson') # Correlation Matrix
	
	f, ax = plt.subplots(figsize=(11, 9))
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(10, 275, as_cmap=True)
	
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr,
		cmap=cmap,
		square=True, 
		xticklabels=True, 
		yticklabels=True,
		linewidths=.5, 
		cbar_kws={"shrink": .5}, 
		ax=ax)

	plt.title("Pearson Correlation Matrix")
	plt.yticks(rotation = 0)
	plt.xticks(rotation = 270)
	
	plt.show()
	plt.close()

	# BoxPlot
	pltBoxPlot(-.05, 50, breastCancer, 'Pre-Processed')

	# Visuals relating to normalized data to show significant difference
	breastCancerNorm = normalize_df(breastCancer)
			
	print("Here's our newly transformed data: \n", 
		breastCancerNorm.head())
	print("Describe function with transformed data: \n", 
		breastCancerNorm.describe())

	pltBoxPlot(-.05, 1.05, breastCancerNorm, 'Transformed')

# CREATE APPROPRIATE SETS FOR MODELING

training_set, class_set, test_set, test_class_set = splitSets(breastCancer)

breastCancerNorm = normalize_df(breastCancer)
training_set_scaled, class_set_scaled, test_set_scaled, \
test_class_set_scaled = splitSets(breastCancerNorm)

	############################################
	##    RUNNING MACHINE LEARNING MODELS     ##
	############################################

def kthNearestNeighbor():
	'''
	Function Kth Nearest Neighbor using k=9
	by running (on terminal):

	# KNN Optimal K
	# Inspired by: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
	myKs = []
	for i in range(0, 50):
		if (i % 2 != 0):
			myKs.append(i)	
	cross_vals = []
	for k in myKs:
		knn = KNeighborsClassifier(n_neighbors=k)
		scores = cross_val_score(knn,
		training_set, 
		class_set['diagnosis'], 
		cv = 10, 
		scoring='accuracy')
		cross_vals.append(scores.mean())
	MSE = [1 - x for x in cross_vals]
	optimal_k = myKs[MSE.index(min(MSE))]
	print("Optimal K is {0}".format(optimal_k))
	
	$ python breastCancerWisconsinDataSet_MachineLearning.py KNN 
	'''

	print('''
	#################################
	## FITTING MODEL KNN USING k=7 ##
	#################################
	'''
	)

	fit_KNN = KNeighborsClassifier(n_neighbors=7)
	fit_KNN.fit(training_set, 
		class_set['diagnosis'])
	
	print(fit_KNN)
	
	print('''
	###############################
	## TRAINING SET CALCULATIONS ##
	###############################
	'''
	)

	# We predict the class for our training set
	predictionsTrain = fit_KNN.predict(training_set) 
	
	# Here we create a matrix comparing the actual values 
	# vs. the predicted values
	print(pd.crosstab(predictionsTrain, 
		class_set['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))
	
	# Measure the accuracy based on the trianing set
	accuracyTrain = fit_KNN.score(training_set, 
		class_set['diagnosis'])
	
	print("Here is our accuracy for our training set:\n {0: .3f}"\
		.format(accuracyTrain))

	train_error_rate = 1 - accuracyTrain  
	print("The train error rate for our model is:\n {0: .3f}"\
		.format(train_error_rate))

	print('''
	###############################
	##      CROSS VALIDATION     ##
	###############################
	'''
	)

	crossVD(fit_KNN, test_set, test_class_set['diagnosis'])
	

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)

	# First we predict the Dx for the test set and call it predictions
	predictions = fit_KNN.predict(test_set)
	
	# Let's compare the predictions vs. the actual values
	print(pd.crosstab(predictions, 
		test_class_set['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))
	
	# Let's get the accuracy of our test set
	accuracy = fit_KNN.score(test_set, 
		test_class_set['diagnosis'])
	
	# TEST ERROR RATE!!
	print("Here is our accuracy for our test set:\n {0: .3f}"\
		.format(accuracy))

	# Here we calculate the test error rate!
	test_error_rate = 1 - accuracy
	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate))

	# ROC Curve and AUC Calculations
	fpr, tpr, _ = roc_curve(predictions, 
		test_class_set)

	auc_knn = auc(fpr, tpr)

	# ROC Curve
	# NOTE: These functions were created in the helper.py 
	# script to reduce lines of code
	# refer to helper.py for additional information
	plotROC(fpr, tpr, auc_knn, 0)

	# Zoomed in ROC Curve
	plotROCZoom(fpr, tpr, auc_knn, 0)
	
def randomForest():
	'''
	Function performs a random forest 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py RF 
	'''
	fit_RF = RandomForestClassifier(random_state = 42, 
		criterion='entropy',
		n_estimators = 500,
		max_features = 4,
		bootstrap=True)

	fit_RF.fit(training_set, 
		class_set['diagnosis'])

	print('''
	####################################################
	##          FITTING MODEL USING 500 TREES         ##
	####################################################
	'''
	)

	print(fit_RF)

	print('''
	###############################
	##    VARIABLE IMPORTANCE    ##
	###############################
	'''
	)

	importancesRF = fit_RF.feature_importances_
	indicesRF = np.argsort(importancesRF)[::-1]



	varImport(namesInd, importancesRF, indicesRF)

	indRf = sorted(importancesRF) # Sort by Decreasing order
	index = np.arange(30)

	feature_space = []
	for i in range(29, -1, -1):
		feature_space.append(namesInd[indicesRF[i]])

	varImportPlot(index, feature_space, indRf)


	print('''
	############################################
	##      HYPERPARAMETER OPTIMIZATION       ##
	############################################
	'''
	)

	print("Note: Remove commented code to see this section")
	print("chosen parameters: {'criterion': 'entropy', 'max_depth': 4, 'bootstrap': True}\
	 \n elapsed time of estimation: 111.895 seconds")
	#start = time.time()

	#param_dist = {"max_depth": [2, 3, 4],
	#"bootstrap": [True, False],
	#"criterion": ["gini", "entropy"]}

	#gs_rf = GridSearchCV(fit_RF, cv = 10,
		#param_grid=param_dist)

	#gs_rf.fit(training_set, class_set['diagnosis'])
	#print(gs_rf.best_params_)
	#end = time.time()
	#print(end - start)


	print('''
	###############################
	##      CROSS VALIDATION     ##
	###############################
	'''
	)


	# CROSS VALIDATION
	crossVD(fit_RF, test_set, test_class_set['diagnosis'])

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)
	predictions_RF = fit_RF.predict(test_set)
	
	print(pd.crosstab(predictions_RF, 
		test_class_set['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))

	accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])

	print("Here is our mean accuracy on the test set:\n {0: .3f}"\
		.format(accuracy_RF))

	# Here we calculate the test error rate!
	test_error_rate_RF = 1 - accuracy_RF
	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_RF))

	# ROC Curve stuff
	fpr2, tpr2, _ = roc_curve(predictions_RF, 
		test_class_set)
	
	auc_rf = auc(fpr2, tpr2)
	# ROC Curve
	plotROC(fpr2, tpr2, auc_rf, 1)
	# Zoomed in ROC Curve
	plotROCZoom(fpr2, tpr2, auc_rf, 1)

def neuralNetworks():
	'''
	Function performs a neural network 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py NN 
	'''
	fit_NN = MLPClassifier(solver='lbfgs', 
		hidden_layer_sizes=(8, ), 
		activation='logistic',
		learning_rate_init=0.05,
		random_state=7)
	
	print('''
	##################################
	##         FITTING MLP          ##
	##################################
	'''
	)

	fit_NN.fit(training_set_scaled, 
		class_set_scaled['diagnosis'])
	
	print(fit_NN)

	print('''
	############################################
	##       HYPERPARAMETER OPTIMIZATION      ##
	############################################
	'''
	)

	print("Note: Remove commented code to see this section")
	print("chosen parameters: {'learning_rate_init': 0.05, 'hidden_layer_sizes': 8, 'activation': 'logistic'} \
		\n Estimated time: 26.818 seconds")

	#start = time.time()
	#gs = GridSearchCV(fit_NN, cv = 10,
		#param_grid={
		#'learning_rate_init': [0.05, 0.01, 0.005, 0.001],
		#'hidden_layer_sizes': [4, 8, 12],
		#'activation': ["relu", "identity", "tanh", "logistic"]})
	
	#gs.fit(training_set_scaled, class_set_scaled['diagnosis'])
	#print(gs.best_params_)
	#end = time.time()
	#print(end - start)

	print('''
	################################
	##      CROSS VALIDATION      ##
	################################
	'''
	)

	crossVD(fit_NN, test_set_scaled, 
		test_class_set_scaled['diagnosis'])	

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)

	predictions_NN = fit_NN.predict(test_set_scaled)
	
	print(pd.crosstab(predictions_NN, 
		test_class_set_scaled['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))

	accuracy_NN = fit_NN.score(test_set_scaled, 
		test_class_set_scaled['diagnosis'])

	print("Here is our mean accuracy on the test set:\n {0: .3f}"\
		.format(accuracy_NN))

	# Here we calculate the test error rate!
	test_error_rate_NN = 1 - accuracy_NN
	
	print("The test error rate for our model is:\n {0: .3f}"\
		.format(test_error_rate_NN))

	# ROC Curve stuff
	fpr3, tpr3, _ = roc_curve(predictions_NN, test_class_set_scaled)
	auc_nn = auc(fpr3, tpr3)
	# ROC Curve
	plotROC(fpr3, tpr3, auc_nn, 2)

	# Zoomed in ROC Curve
	plotROCZoom(fpr3, tpr3, auc_nn, 2)

if __name__ == '__main__':
	if len(sys.argv) == 2:
		if sys.argv[1] == 'EA':
			exploratoryAnalysis()
		elif sys.argv[1] == 'VEA':
			visualExplorAnalysis()
		elif sys.argv[1] == 'KNN':
			kthNearestNeighbor()
		elif sys.argv[1] == 'RF':
			randomForest()
		elif sys.argv[1] == 'NN':
			neuralNetworks()