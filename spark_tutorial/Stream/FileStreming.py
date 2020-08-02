'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-08-02 20:06:11
@LastEditors: Please set LastEditors
@Description: 
'''
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

conf = SparkConf().setAppName('TestDStream').setMaster('local[2]')
sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 10)

lines = ssc.textFileStream('file:///mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/logfile')
words = lines.flatMap(lambda line: line.split(' '))
wordCounts = words.map(lambda x : (x,1)).reduceByKey(lambda a,b:a+b)
wordCounts.pprint()

ssc.start()
ssc.awaitTermination()