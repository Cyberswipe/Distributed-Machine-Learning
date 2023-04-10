import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
import psutil
import logging
from time import time

# Create a SparkSession
spark = SparkSession.builder.appName("Logistic Regression").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
logging.getLogger("py4j").setLevel(logging.ERROR)

# Load the data as an RDD and skip the header row
train_rdd = sc.textFile("mnist_train.csv").filter(lambda x: "label" not in x).map(lambda x: x.split(","))
test_rdd = sc.textFile("mnist_test.csv").filter(lambda x: "label" not in x).map(lambda x: x.split(","))

# Convert the data to labeled points
train_data = train_rdd.map(lambda x: LabeledPoint(float(x[0]), SparseVector(len(x[1:]), {i: float(x[i+1]) for i in range(len(x[1:]))})))
test_data = test_rdd.map(lambda x: LabeledPoint(float(x[0]), SparseVector(len(x[1:]), {i: float(x[i+1]) for i in range(len(x[1:]))})))

# Train the model without partitioning
start_time = time()
lr = LogisticRegressionWithSGD.train(train_data, iterations=100, regParam=0.01, regType='l2', intercept=True)
end_time = time()
print("Training Time without Partitioning:", end_time-start_time, "seconds")

# Make predictions on the test data and calculate accuracy
predictions = test_data.map(lambda x: (x.label, lr.predict(x.features)))
accuracy = predictions.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Accuracy without Partitioning:", accuracy)

# Train the model with partitioning
start_time_partitioned = time()
train_data_partitioned = train_data.repartition(20)# partition the data
lr_partitioned = LogisticRegressionWithSGD.train(train_data_partitioned, iterations=100, regParam=0.01, regType='l2', intercept=True)
end_time_partitioned = time()
print("Training Time with Partitioning:", end_time-start_time, "seconds")

# Make predictions on the test data and calculate accuracy
predictions_partitioned = test_data.map(lambda x: (x.label, lr_partitioned.predict(x.features)))
accuracy_partitioned = predictions_partitioned.filter(lambda x: x[0] == x[1]).count() / float(test_data.count())
print("Accuracy with Partitioning:", accuracy_partitioned)

# Calculate CPU utilization
cpu_utilization = sum(psutil.cpu_percent(percpu=True))/psutil.cpu_count()

# Print results
print("CPU Utilization:", cpu_utilization, "%")
print("Speedup with Partitioning:", (end_time-start_time)/(end_time_partitioned-start_time_partitioned))
print("Speedup with Partitioning per CPU:", (end_time-start_time)/(end_time_partitioned-start_time_partitioned)/cpu_utilization)

# Stop the SparkSession
spark.stop()
