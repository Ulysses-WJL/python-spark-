'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 17:56:26
LastEditors: Please set LastEditors
@Description:  2.3 版本之后python 无法使用 Streaming + Kafka Integration
'''
from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: KafkaWordCount.py <zk> <topic>", file=sys.stderr)
        exit(-1)
    sc = SparkContext(master='local[2]', appName="PythonStreamingKafkaWordCount")
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc, 2)
    zkQuorum, topic = sys.argv[1:]
    # zkQuorum, topic = sys.argv[1:]
    # kvs = KafkaUtils.createDirectStream(
    #             ssc, ['first'],
    #             kafkaParams={"metadata.broker.list": "127.0.1:9092"})
    kvs = KafkaUtils.createStream(
        ssc,  zkQuorum=zkQuorum, 
        groupId='spark-streaming-consumer', topics={topic:1})
    
    lines = kvs.map(lambda x: x[1])
    counts = lines.flatMap(lambda line: line.split(" "))\
        .map(lambda word: (word, 1))\
        .reduceByKey(lambda a, b: a+b)
    counts.pprint()
    ssc.start()
    ssc.awaitTermination()
