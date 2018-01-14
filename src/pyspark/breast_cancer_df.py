# Load packages
from pyspark.sql.functions import col
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier


rdd = sc.textFile('data/data.txt').map(lambda lines: lines.split(" "))

df = rdd.toDF()

data = df.selectExpr('_1 as label', '_2 as radius_mean', 
	'_3 as texture_mean', '_4 as perimeter_mean', 
	'_5 as area_mean', '_6 as smoothness_mean', 
	'_7 as compactness_mean', '_8 as concavity_mean', 
	'_9 as concave_points_mean', '_10 as symmetry_mean', 
	'_11 as fractal_dimension_mean', '_12 as radius_se', 
	'_13 as texture_se', '_14 as perimeter_se', 
	'_15 as area_se', '_16 as smoothness_se', 
	'_17 as compactness_se', '_18 as concavity_se', 
	'_19 as concave_points_se', '_20 as symmetry_se', 
	'_21 as fractal_dimension_se', '_22 as radius_worst', 
	'_23 as texture_worst', '_24 as perimeter_worst', 
	'_25 as area_worst', '_26 as smoothness_worst', 
	'_27 as compactness_worst', '_28 as concavity_worst', 
	'_29 as concave_points_worst', '_30 as symmetry_worst', 
	'_31 as fractal_dimension_worst')


# Converting to correct data types
newData = data.select([col(c).cast('float') if c != 'label' else col(c).cast('int') for c in data.columns ])

# For loops to output the describe functionality neatly 
mylist = []
mylist2 = []
for i in range(0, 31):
    if (i % 2 != 0):
    	mylist.append(newData.columns[i])
    else:
    	mylist2.append(newData.columns[i])

# Now we use the newly created lists that have even and odd columns respectively
# to see some basic statistics for our dataset
for i in range(0, 15): 	
	newData.describe(mylist[i], mylist2[i]).show()

# Important meta-data inputting for when I start running models!
# Meta-data for the feature space
featureIndexer = VectorAssembler(
	inputCols = [x for x in newData.columns if x != 'label'],
	outputCol = 'features')

df = featureIndexer.transform(newData)

# Some tests to see if things came out properly
df.select(df['features']).show()
df.select(df['label']).show()

# Creating training and test sets
(trainingSet, testSet) = df.randomSplit([0.7, 0.3])

####################
## DECISION TREES ##
####################

# Creating training and test sets

dt = DecisionTreeClassifier(labelCol="label",
	featuresCol = "features")

#pipeline_dt = Pipeline(stages=[labelIndexer0, featureIndexer0, dt])

model_dt = dt.fit(trainingSet)

predictions_dt = model_dt.transform(testSet)

# Select example rows to display.
predictions_dt.select("prediction", 
	"label", 
	"features").show(5)

# Select (prediction, true label) and compute test error
evaluator_dt = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="accuracy")

accuracy_dt = evaluator_dt.evaluate(predictions_dt)

print("Test Error = %g " % (1.0 - accuracy_dt))
'''
Test Error = 0.0697674 
'''

#########################
## Random Forest Model ##
#########################

rf = RandomForestClassifier(labelCol='label',
	maxDepth=4,
	impurity="gini",
	numTrees=500,
	seed=42)

model_rf = rf.fit(trainingSet)

predictions_rf = model_rf.transform(testSet)

predictions_rf.select("prediction", "label", "features").show(10)

'''
+----------+-----+--------------------+
|prediction|label|            features|
+----------+-----+--------------------+
|       0.0|    0|[0.0,0.1258031932...|
|       0.0|    0|[0.05859245005374...|
|       0.0|    0|[0.07652986773450...|
|       0.0|    0|[0.07747645570059...|
|       0.0|    0|[0.07998483256627...|
|       0.0|    0|[0.09025507729212...|
|       0.0|    0|[0.09318944582402...|
|       0.0|    0|[0.11756354432107...|
|       0.0|    0|[0.11940932766481...|
|       0.0|    0|[0.13280324046146...|
+----------+-----+--------------------+
only showing top 10 rows
'''

evaluator_rf = MulticlassClassificationEvaluator(labelCol="label", 
	predictionCol="prediction", 
	metricName="accuracy")

accuracy_rf = evaluator_rf.evaluate(predictions_rf)
print("Test Error = %g" % (1.0 - accuracy_rf))
'''
Test Error = 0.0223
'''

#####################
## NEURAL NETWORKS ##
#####################

########################
## RESCALING DATA SET ##
########################
# Typically for Neural Networks to perform better 
# a lot of preprocessing has to go into the data
# So I scaled the feature space to have min = 0 and max = 1

scaler = MinMaxScaler(inputCol='features', outputCol='scaledFeatures')

scalerModel = scaler.fit(df)

scaledData = scalerModel.transform(df)

print("Features scaled to range: [%f, %f]" % (scaler.getMin(), scaler.getMax()))

scaledData.select("features", "scaledFeatures").show()

new_df = scaledData.selectExpr("label", "radius_mean", "texture_mean", 
	"perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
	 "concavity_mean", "concave_points_mean", "symmetry_mean", 
	 "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", 
	 "area_se", "smoothness_se", "compactness_se", "concavity_se", 
	 "concave_points_se", "symmetry_se", "fractal_dimension_se", 
	 "radius_worst", "texture_worst", "perimeter_worst", 
	 "area_worst", "smoothness_worst", "compactness_worst", 
	 "concavity_worst", "concave_points_worst", "symmetry_worst", 
	 "fractal_dimension_worst","features as oldFeature", 
	 "scaledFeatures as features")

# Creating training and test sets
(trainingSet_scaled, testSet_scaled) = new_df\
.randomSplit([0.7, 0.3])

layers = [30, 5, 4, 2]

trainer = MultilayerPerceptronClassifier(maxIter=100, 
	layers=layers, 
	blockSize=128, 
	seed=1234)

model_nn = trainer.fit(trainingSet_scaled)

result_nn = model_nn.transform(testSet_scaled)
predictions_nn = result_nn.select("prediction", "label")
evaluator_nn = MulticlassClassificationEvaluator(metricName="accuracy")

accuracy_nn = evaluator_nn.evaluate(predictions_nn) 

print("Test Error = %g" % (1.0 - accuracy_nn))
'''
Test Error = 0.0314465
'''