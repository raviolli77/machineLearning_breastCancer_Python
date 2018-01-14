# LOAD APPROPRIATE PACKAGE
import numpy as np
from pyspark.context import SparkContext 
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext.getOrCreate()
data = MLUtils.loadLibSVMFile(sc, 'data/dataLibSVM.txt')
print(data)
# NEXT LET'S CREATE THE APPROPRIATE TRAINING AND TEST SETS
# WE'LL BE SETTING THEM AS 70-30, ALONG WITH SETTING A
# RANDOM SEED GENERATOR TO MAKE MY RESULTS REPRODUCIBLE  

(trainingSet, testSet) = data.randomSplit([0.7, 0.3], seed = 7)

##################
# DECISION TREES #
##################

fitDT = DecisionTree.trainClassifier(trainingSet, 
	numClasses=2, 
	categoricalFeaturesInfo={},
	impurity='gini', 
	maxDepth=3, 
	maxBins=32)

print(fitDT.toDebugString())

predictionsDT = fitDT.predict(testSet.map(lambda x: x.features))

labelsAndPredictionsDT = testSet.map(lambda lp: lp.label).zip(predictionsDT)

# Test Error Rate Evaluations

testErrDT = labelsAndPredictionsDT.filter(lambda (v, p): v != p).count() / float(testSet.count())

print('Test Error = {0}'.format(testErrDT))

# Instantiate metrics object
metricsDT = BinaryClassificationMetrics(labelsAndPredictionsDT)

# Area under ROC curve
print("Area under ROC = {0}".format(metricsDT.areaUnderROC))

#################
# RANDOM FOREST #
#################

fitRF = RandomForest.trainClassifier(trainingSet, 
	numClasses = 2, 
	categoricalFeaturesInfo = {},
	numTrees = 500,
	featureSubsetStrategy="auto",
	impurity = 'gini', # USING GINI INDEX FOR OUR RANDOM FOREST MODEL 
	maxDepth = 4,
	maxBins = 100)

predictionsRF = fitRF.predict(testSet.map(lambda x: x.features))

labelsAndPredictions = testSet.map(lambda lp: lp.label).zip(predictionsRF)


testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testSet.count())

print('Test Error = {0}'.format(testErr))
print('Learned classification forest model:')
print(fitRF.toDebugString())

# Instantiate metrics object
metricsRF = BinaryClassificationMetrics(labelsAndPredictions)

# Area under ROC curve
print("Area under ROC = {0}".format(metricsRF.areaUnderROC))

###################
# NEURAL NETWORKS #
###################

# See Scala Script