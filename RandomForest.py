import findspark
findspark.init()
import logging
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
import psutil
from time import time

conf = SparkConf().setAppName("Random Forest").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName("Random Forest").getOrCreate()
sc.setLogLevel("ERROR")


train_rdd = sc.textFile("mnist_train.csv").filter(lambda x: "label" not in x).map(lambda x: x.split(","))
test_rdd = sc.textFile("mnist_test.csv").filter(lambda x: "label" not in x).map(lambda x: x.split(","))

train_data = train_rdd.map(lambda x: LabeledPoint(float(x[0]), SparseVector(len(x[1:]), {i: float(x[i+1]) for i in range(len(x[1:]))})))
test_data = test_rdd.map(lambda x: LabeledPoint(float(x[0]), SparseVector(len(x[1:]), {i: float(x[i+1]) for i in range(len(x[1:]))})))


start_time = time()
model = RandomForest.trainClassifier(train_data, numClasses=10, categoricalFeaturesInfo={}, numTrees=50, featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32)
end_time = time()
print("Training Time without Partitioning:", end_time-start_time, "seconds")


predictions = model.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda lp: lp.label).zip(predictions)
accuracy = labels_and_predictions.filter(lambda lp: lp[0] == lp[1]).count() / float(test_data.count())
print("Accuracy without Partitioning:", accuracy)


print("CPU utilization without Partitioning: ", psutil.cpu_percent(interval=1), "%")

spark.stop()
