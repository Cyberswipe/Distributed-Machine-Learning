import findspark
findspark.init()
import logging
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkContext
from time import time
import psutil
import os

sc = SparkContext(appName="Logistic Regression")
sc.setLogLevel("ERROR")
logging.getLogger("py4j").setLevel(logging.ERROR)

train_rdd = sc.textFile("mnist_train.csv").filter(lambda x: "label" not in x).map(lambda x: x.split(","))
test_rdd = sc.textFile("mnist_test.csv").filter(lambda x: "label" not in x).map(lambda x: x.split(","))
train_data = train_rdd.map(lambda x: LabeledPoint(float(x[0]), SparseVector(len(x[1:]), {i: float(x[i+1]) for i in range(len(x[1:]))})))
test_data = test_rdd.map(lambda x: LabeledPoint(float(x[0]), SparseVector(len(x[1:]), {i: float(x[i+1]) for i in range(len(x[1:]))})))


train_data = train_data.repartition(3)


from pyspark.mllib.classification import LogisticRegressionWithLBFGS
start_time = time()
model = LogisticRegressionWithLBFGS.train(train_data, numClasses=10, iterations=100, regParam=0.01, regType='l2')
end_time = time()
print("Training Time with Partitioning:", end_time-start_time, "seconds")
print("CPU usage during training with partitioning:", psutil.cpu_percent())
print("CPU Utilization (workers):", psutil.cpu_percent(percpu=True))
print("Memory Utilization:", psutil.virtual_memory())

predictions = model.predict(test_data.map(lambda x: x.features))
labels_and_predictions = test_data.map(lambda lp: lp.label).zip(predictions)
accuracy = labels_and_predictions.filter(lambda lp: lp[0] == lp[1]).count() / float(test_data.count())
print("Accuracy with Partitioning:", accuracy)

