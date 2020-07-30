'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-19 20:16:45
@LastEditors: your name
@Description: 
'''
from pyspark import SparkContext
logFile = "file:///usr/local/spark/README.md"  
sc = SparkContext("local", "first app")
logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()
print("Lines with a: {}, lines with b: {}".format(numAs, numBs))
