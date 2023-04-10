import findspark
findspark.init()
import logging
from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from time import time
import psutil
import os


spark = SparkSession.builder.appName("Logistic Regression").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")
logging.getLogger("py4j").setLevel(logging.ERROR)


# Load the data as a DataFrame
train_df = spark.read.csv("mnist_train.csv", header=True, inferSchema=True)
test_df = spark.read.csv("mnist_test.csv", header=True, inferSchema=True)

train_df = train_df.rdd.map(lambda x: (x[0], Vectors.dense(x[1:])))
train_df = spark.createDataFrame(train_df, ["label", "features"])
test_df = test_df.rdd.map(lambda x: (x[0], Vectors.dense(x[1:])))
test_df = spark.createDataFrame(test_df, ["label", "features"])

# Train the model without partitioning
start_time = time()
lr = LogisticRegression(maxIter=50, regParam=0.01, elasticNetParam=0.8)
model = lr.fit(train_df)
end_time = time()
print("Training Time without Partitioning:", end_time-start_time, "seconds")
# CPU monitor
print("CPU usage during training without partitioning:", psutil.cpu_percent())


predictions = model.transform(test_df)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_df.count())
print("Accuracy without Partitioning:", accuracy)

# Print the gradient and loss at each iteration
training_summary = model.summary
print("Gradient per iteration without Partitioning:")
for i, grad in enumerate(training_summary.objectiveHistory):
    print("Iteration", i, ": Gradient =", grad)

print("Loss at each iteration without partitioning:")
for i, obj in enumerate(training_summary.objectiveHistory):
    print("Iteration %d: Loss = %.6f" % (i, obj))

# Train the model with partitioning

start_time = time()
train_df_partitioned = train_df.repartition(40)
num_partitions = 5
train_df_partitioned = train_df.coalesce(num_partitions)
lr_partitioned = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.8)
model_partitioned = lr_partitioned.fit(train_df_partitioned)
end_time = time()
print("Training Time with Partitioning:", end_time-start_time, "seconds")

print("CPU usage during training with partitioning:", psutil.cpu_percent())

predictions_partitioned = model_partitioned.transform(test_df)
accuracy_partitioned = predictions_partitioned.filter(predictions_partitioned.label == predictions_partitioned.prediction).count() / float(test_df.count())
print("Accuracy with Partitioning:", accuracy_partitioned)

training_summary_partitioned = model_partitioned.summary
print("Gradient per iteration with Partitioning:")
for i, grad in enumerate(training_summary_partitioned.objectiveHistory):
    print("Iteration", i, ": Gradient =", grad)

print("Loss at each iteration with partitioning:")
for i, obj in enumerate(training_summary_partitioned.objectiveHistory):
    print("Iteration %d: Loss = %.6f" % (i, obj))

Stop the SparkSession
spark.stop()
