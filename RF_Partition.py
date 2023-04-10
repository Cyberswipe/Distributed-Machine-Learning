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


for num_partitions in [1, 3, 5]:
    start_time = time()
    train_data_partitioned = train_data.repartition(num_partitions)
    model_partitioned = RandomForest.trainClassifier(train_data_partitioned, numClasses=10, categoricalFeaturesInfo={}, numTrees=50, featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32)
    end_time = time()
    print("Training Time with", num_partitions, "partitions:", end_time-start_time, "seconds")

    # Make predictions on the test data and calculate accuracy
    predictions_partitioned = model_partitioned.predict(test_data.map(lambda x: x.features))
    labels_and_predictions_partitioned = test_data.map(lambda lp: lp.label).zip(predictions_partitioned)
    accuracy_partitioned = labels_and_predictions_partitioned.filter(lambda lp: lp[0] == lp[1]).count() / float(test_data.count())
    print("Accuracy with", num_partitions, "partitions:", accuracy_partitioned)

    # CPU utilization with partitioning
    print("CPU utilization with", num_partitions, "partitions: ", psutil.cpu_percent(interval=1), "%\n")
    print("Memory utilization with", num_partitions, "partitions: ", psutil.virtual_memory())

# Stop the SparkSession
