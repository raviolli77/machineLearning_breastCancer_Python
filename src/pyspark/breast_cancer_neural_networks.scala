// Load appropriate packages
// Neural Networks
// Compatible with Apache Zeppelin
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

// Read in file
val data = spark.read.format("libsvm")
  .load("data/data.txt")

data.collect()

// Pre-processing 
val scaler = new MinMaxScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")

val scalerModel = scaler.fit(data)

val scaledData = scalerModel.transform(data)
println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
scaledData.select("features", "scaledFeatures").show()

// Changing RDD files variable names to get accurate predictions
val newNames = Seq("label", "oldFeatures", "features")
val data2 = scaledData.toDF(newNames: _*)

val splits = data2.randomSplit(Array(0.7, 0.3), seed = 1234L)
val trainingSet = splits(0)
val testSet = splits(1)

trainingSet.select("label", "features").show(25)

// Neural Networks
val layers = Array[Int](30, 5, 4, 2)

// Train the Network
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

val fitNN = trainer.fit(trainingSet)

// Predict the Test set 
val results = fitNN.transform(testSet)
val predictionAndLabelsNN = results.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println("Test error rate = " + (1 - evaluator.evaluate(predictionAndLabelsNN)))

println("Test set accuracy = " + evaluator.evaluate(predictionAndLabelsNN))