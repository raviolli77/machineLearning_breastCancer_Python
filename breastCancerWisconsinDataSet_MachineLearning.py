#!/usr/bin/env python3

# WISCONSIN BREAST CANCER MACHINE LEARNING (PYTHON)

# Project by Raul Eulogio
import sys
import numpy as np
import pandas as pd # Data frames
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
from sklearn.model_selection import train_test_split # Create training and test sets
from sklearn.model_selection import KFold, cross_val_score # Cross validation
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.neural_network import MLPClassifier # Neural Networks
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # Calculating Area Under Curve for ROC's!
from urllib.request import urlopen # Get data from UCI Machine Learning Repository

pd.set_option('display.max_columns', 500) # Included to show all the columns 
# since it is a fairly large data set
plt.style.use('ggplot') # Using ggplot2 style visuals because that's how I learned my visuals 
# and I'm sticking to it!

	#################################
	##        LOADING DATA         ##
	#################################

UCI_data_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'

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
		elif item not in frame.select_dtypes(include=[np.float]):
			breastCancerNorm[item] = frame[item]
	return breastCancerNorm


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
	print("Distribution of Diagnoses:")
	print("The percentage of Malignant Dx is: {0:.2f}%".format(perMal)) 
	print("The percentage of Begnin Dx is: {0:.2f}%".format(perBeg))

def exploratoryAnalysis():
	'''
	Function shows various statistical calculations done as a preliminary exploratory analysis 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py EA 
	'''
	print('''
	#################################
	##    EXPLORATORY ANALYSIS     ##
	#################################
	''')
	
	print('''
	########################################
	##    DATA FRAME SHAPE AND DTYPES     ##
	########################################
	''')
	print("Here's the dimensions of our data frame:\n", breastCancer.shape)
	print("Here's the data types of our columns:\n", breastCancer.dtypes)
	
	print("Some more statistics for our data frame: ", breastCancer.describe())

	print('''
	##########################################
	##      STATISTICS RELATING TO DX       ##
	##########################################
	''')
	# Let's look at the count of the new representations of our Dx's
	print("Count of the Dx:\n", breastCancer['diagnosis'].value_counts())

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
	breastCancerSamp = breastCancer.loc[:, 
                                    	['concave_points_worst', 'concavity_mean', 
                                     	'perimeter_worst', 'radius_worst', 
                                     	'area_worst', 'diagnosis']]
	
	sns.set_palette(palette = ('Red', '#875FDB'))
	pairPlots = sns.pairplot(breastCancerSamp, hue = 'diagnosis')
	pairPlots.set(axis_bgcolor='#fafafa')
	
	plt.show()
	plt.close()

	# Pearson Correlation Matrix
	corr = breastCancer.corr(method = 'pearson') # Correlation Matrix
	
	f, ax = plt.subplots(figsize=(11, 9))
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(10, 275, as_cmap=True)
	
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr,  cmap=cmap,square=True, 
            	xticklabels=True, yticklabels=True,
            	linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
	plt.yticks(rotation = 0)
	plt.xticks(rotation = 270)
	
	plt.show()
	plt.close()

	# BoxPlot
	f, ax = plt.subplots(figsize=(11, 15))
	
	ax.set_axis_bgcolor('#fafafa')
	ax.set(xlim=(-.05, 50))
	plt.ylabel('Dependent Variables')
	plt.title("Box Plot of Pre-Processed Data Set")
	ax = sns.boxplot(data = breastCancer, orient = 'h', palette = 'Set2')
	
	plt.show()
	plt.close()

	# Visuals relating to normalized data to show significant difference
	normalize_df(breastCancer)
			
	print("Here's our newly transformed data: \n", breastCancerNorm.head())
	print("Describe function with transformed data: \n", breastCancerNorm.describe())

	f, ax = plt.subplots(figsize=(11, 15))

	ax.set_axis_bgcolor('#fafafa')
	plt.title("Box Plot of Transformed Data Set (Breast Cancer Wisconsin Data Set)")
	ax.set(xlim=(-.05, 1.05))
	ax = sns.boxplot(data = breastCancerNorm[1:29], orient = 'h', palette = 'Set2')

	plt.show()
	plt.close()


	############################################
	##    CREATING TRAINING AND TEST SETS     ##
	############################################

# Since the last frame was only within the local function 
# We create the normalized data frame again
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


	############################################
	##    RUNNING MACHINE LEARNING MODELS     ##
	############################################

def kthNearestNeighbor():
	'''
	Function Kth Nearest Neighbor using k=9
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py KNN 
	'''

	print('''
	#################################
	## FITTING MODEL KNN USING k=9 ##
	#################################
	'''
	)

	breastCancerKnn = KNeighborsClassifier(n_neighbors=9)
	breastCancerKnn.fit(training_set, class_set['diagnosis'])
	print(breastCancerKnn)
	
	print('''
	###############################
	## TRAINING SET CALCULATIONS ##
	###############################
	'''
	)
	# We predict the class for our training set
	predictionsTrain = breastCancerKnn.predict(training_set) 
	
	# Here we create a matrix comparing the actual values vs. the predicted values
	print(pd.crosstab(predictionsTrain, class_set['diagnosis'], 
                  	rownames=['Predicted Values'], colnames=['Actual Values']))
	
	# Measure the accuracy based on the trianing set
	accuracyTrain = breastCancerKnn.score(training_set, class_set['diagnosis'])
	
	print("Here is our accuracy for our training set:\n",
		'%.3f' % (accuracyTrain * 100), '%')

	train_error_rate = 1 - accuracyTrain  
	print("The train error rate for our model is:\n",
		'%.3f' % (train_error_rate * 100), '%')
	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)

	# First we predict the Dx for the test set and call it predictions
	predictions = breastCancerKnn.predict(test_set)
	
	# Let's compare the predictions vs. the actual values
	print(pd.crosstab(predictions, test_class_set['diagnosis'], 
					rownames=['Predicted Values'], 
					colnames=['Actual Values']))
	
	# Let's get the accuracy of our test set
	accuracy = breastCancerKnn.score(test_set, test_class_set['diagnosis'])
	
	# TEST ERROR RATE!!
	print("Here is our accuracy for our test set:\n",
		'%.3f' % (accuracy * 100), '%')

	# Here we calculate the test error rate!
	test_error_rate = 1 - accuracy
	print("The test error rate for our model is:\n",
		'%.3f' % (test_error_rate * 100), '%')
	# ROC Curve and AUC Calculations
	fpr, tpr, _ = roc_curve(predictions, test_class_set)

	auc_knn = auc(fpr, tpr)

	# ROC Curve
	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr, tpr, label='Kth-NN ROC Curve  (area = %.4f)' % auc_knn, 
		color = 'deeppink', 
		linewidth=1)

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve For Kth Nearest Neighbor (AUC = {0: 0.3f})'.format(auc_knn))
	
	plt.show()
	plt.close()

	# Zoomed in
	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr, tpr, label='Kth-NN ROC Curve  (area = %.4f)' % auc_knn, 
		color = 'deeppink', 
		linewidth=1)

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.001, 0.2])
	plt.ylim([0.7, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve For Kth Nearest Neighbor (AUC = {0: 0.3f})'.format(auc_knn))
	
	plt.show()
	plt.close()
	#return fpr, tpr, predictions, auc_knn

def decisionTree():
	'''
	Function performs a decision tree 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py DT 
	'''
	print('''
	####################################################
	##       FITTING MODEL USING MAX DEPTH OF 3       ##
	####################################################
	'''
	)
	dt = DecisionTreeClassifier(random_state = 42, 
							criterion='gini', 
							max_depth=3)

	fit = dt.fit(training_set, class_set)
	print(fit)

	print('''
	###############################
	##    VARIABLE IMPORTANCE    ##
	###############################
	'''
	)
	namesInd = names[2:] # Cus the name list has 'id_number' and 'diagnosis' so we exclude those
	
	with open('dotFiles/breastCancerWD.dot', 'w') as f:
		f = export_graphviz(fit, out_file = f,
							feature_names=namesInd,
							rounded = True)

	# Variable Importance for model
	importances = fit.feature_importances_
	indices = np.argsort(importances)[::-1]


	# Print the feature ranking
	print("Feature ranking:")
	
	for f in range(30):
		i = f
		print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
																	namesInd[indices[i]], 
																	importances[indices[f]]))
	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)

	accuracy_dt = fit.score(test_set, test_class_set['diagnosis'])
	
	print("Here is our mean accuracy on the test set:\n",
		'%.2f' % (accuracy_dt * 100), '%')

	predictions_DT = fit.predict(test_set)
	
	print("Table comparing actual vs. predicted values for our test set:")
	print(pd.crosstab(predictions_DT, test_class_set['diagnosis'], 
					rownames=['Predicted Values'], 
					colnames=['Actual Values']))

	# Here we calculate the test error rate!
	test_error_rate_dt = 1 - accuracy_dt
	print("The test error rate for our model is:\n",
		'%.3f' % (test_error_rate_dt * 100) , '%')
	# ROC Curve stuff
	fpr1, tpr1, _ = roc_curve(predictions_DT, test_class_set)

	auc_dt = auc(fpr1, tpr1)

	# ROC Curve
	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr1, tpr1, label='Decision Tree ROC Curve  (area = %.4f)' % auc_dt, 
		color = 'navy', 
		linewidth=1)
	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.01, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve For Decision Tree (AUC = {0: 0.3f})'.format(auc_dt))
	
	plt.show()
	plt.close()

	# Zoomed in Plots
	fig, ax = plt.subplots(figsize=(10, 10))
	plt.plot(fpr1, tpr1, label='Decision Tree ROC Curve  (area = %.4f)' % auc_dt, 
		color = 'navy', 
		linewidth=1)
	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
	
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.001, 0.2])
	plt.ylim([0.7, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve For Decision Tree (AUC = {0: 0.3f})'.format(auc_dt))
	
	plt.show()
	plt.close()
	#return fpr1, tpr1, predictions_DT, auc_dt

def randomForest():
	'''
	Function performs a random forest 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py RF 
	'''
	fit_RF = RandomForestClassifier(random_state = 42, 
		criterion='gini',
		n_estimators = 500,
		max_features = 5)
	fit_RF.fit(training_set, class_set['diagnosis'])
	print('''
	####################################################
	##          FITTING MODEL USING 500 TREES         ##
	####################################################
	'''
	)
	print(fit_RF)

	namesInd = names[2:] # Cus the name list has 'id_number' and 'diagnosis' so we exclude those
	importancesRF = fit_RF.feature_importances_
	indicesRF = np.argsort(importancesRF)[::-1]

	print('''
	###############################
	##    VARIABLE IMPORTANCE    ##
	###############################
	'''
	)
	# Print the feature ranking
	print("Feature ranking:")

	for f in range(30):
		i = f
		print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
			namesInd[indicesRF[i]], 
			importancesRF[indicesRF[f]]))

	indRf = sorted(importancesRF) # Sort by Decreasing order
	index = np.arange(30)

	# PLOTTING VARIABLE IMPORTANCE
	f, ax = plt.subplots(figsize=(11, 11))
	
	ax.set_axis_bgcolor('#fafafa')
	plt.title('Feature importances for Random Forest Model')
	plt.barh(index, indRf,
		align="center", 
		color = '#875FDB')
	plt.yticks(index, ('smoothness_se', 'symmetry_mean', 'texture_se', 
		'concave_points_se', 'fractal_dimension_mean', 
		'symmetry_se', 'compactness_se', 'fractal_dimension_worst', 
		'fractal_dimension_se', 'smoothness_mean', 'concavity_se', 
		'perimeter_se', 'compactness_mean', 'smoothness_worst', 'symmetry_worst', 
		'compactness_worst', 'texture_mean', 'radius_se', 'texture_worst', 'area_se', 
		'concavity_worst', 'area_mean', 'perimeter_mean', 'radius_mean', 'concavity_mean', 
		'radius_worst', 'perimeter_worst', 'concave_points_mean', 
		'area_worst', 'concave_points_worst'))
	
	plt.ylim(-1, 30)
	plt.xlim(0, 0.15)
	plt.xlabel('Gini Importance')
	plt.ylabel('Feature')
	
	plt.show()
	plt.close()

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)
	predictions_RF = fit_RF.predict(test_set)
	print(pd.crosstab(predictions_RF, test_class_set['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))

	accuracy_RF = fit_RF.score(test_set, test_class_set['diagnosis'])

	print("Here is our mean accuracy on the test set:\n",
		'%.3f' % (accuracy_RF * 100), '%')

	# Here we calculate the test error rate!
	test_error_rate_RF = 1 - accuracy_RF
	print("The test error rate for our model is:\n",
		'%.3f' % (test_error_rate_RF * 100), '%')

	# ROC Curve stuff
	fpr2, tpr2, _ = roc_curve(predictions_RF, test_class_set)
	
	auc_rf = auc(fpr2, tpr2)
	# ROC Curve
	fig, ax = plt.subplots(figsize=(10, 10))

	plt.plot(fpr2, tpr2, 
		color = 'red', 
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
	plt.title('ROC Curve For Random Forest (AUC = {0: 0.3f})'.format(auc_rf))

	plt.show()
	plt.close()

	# Zoomed in Plots
	fig, ax = plt.subplots(figsize=(10, 10))

	plt.plot(fpr2, tpr2, 
		color = 'red', 
		linestyle=':', 
		linewidth=4)

	ax.set_axis_bgcolor('#fafafa')
	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
	plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
	plt.xlim([-0.001, 0.2])
	plt.ylim([0.7, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve For Random Forest (AUC = {0: 0.3f})'.format(auc_rf))

	plt.show()
	plt.close()
	#return fpr2, tpr2, predictions_RF, auc_rf

def neuralNetworks():
	'''
	Function performs a neural network 
	by running (on terminal):
	$ python breastCancerWisconsinDataSet_MachineLearning.py NN 
	'''
	fit_NN = MLPClassifier(solver='lbfgs', 
		hidden_layer_sizes=(5, ), 
		random_state=7)

	fit_NN.fit(training_set, class_set['diagnosis'])

	print('''
	####################################################
	##          FITTING MODEL _ HIDDEN LAYERS         ##
	####################################################
	'''
	)

	print(fit_NN)

	print('''
	###############################
	##   TEST SET CALCULATIONS   ##
	###############################
	'''
	)

	predictions_NN = fit_NN.predict(test_set)
	
	print(pd.crosstab(predictions_NN, test_class_set['diagnosis'], 
		rownames=['Predicted Values'], 
		colnames=['Actual Values']))

	accuracy_NN = fit_NN.score(test_set, test_class_set['diagnosis'])

	print("Here is our mean accuracy on the test set:\n", 
		'%.2f' % (accuracy_NN * 100), '%')

	# Here we calculate the test error rate!
	test_error_rate_NN = 1 - accuracy_NN
	print("The test error rate for our model is:\n", 
		'%.3f' % (test_error_rate_NN * 100), '%')

	# ROC Curve stuff
	fpr3, tpr3, _ = roc_curve(predictions_NN, test_class_set)

	auc_nn = auc(fpr3, tpr3)
	#ROC Curve
	fig, ax = plt.subplots(figsize=(10, 10))

	plt.plot(fpr3, tpr3, 
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
	plt.title('ROC Curve For Neural Network (AUC = {0: 0.3f})'.format(auc_nn))

	plt.show()
	plt.close()

	# Zoomed in Plot
	fig, ax = plt.subplots(figsize=(10, 10))

	plt.plot(fpr3, tpr3, 
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
	plt.title('ROC Curve For Neural Network (AUC = {0: 0.3f})'.format(auc_nn))	

	plt.show()
	plt.close()
	#return fpr3, tpr3, predictions_NN, auc_nn

if __name__ == '__main__':
	if len(sys.argv) == 2:
		if sys.argv[1] == 'EA':
			exploratoryAnalysis()
		elif sys.argv[1] == 'VEA':
			visualExplorAnalysis()
		elif sys.argv[1] == 'KNN':
			kthNearestNeighbor()
		elif sys.argv[1] == 'DT':
			decisionTree()
		elif sys.argv[1] == 'RF':
			randomForest()
		elif sys.argv[1] == 'NN':
			neuralNetworks()