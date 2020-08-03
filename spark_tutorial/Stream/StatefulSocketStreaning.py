'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 11:34:32
LastEditors: Please set LastEditors
Description: 
'''
import sys

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: WindowedSocketStreaming.py <hostname> <port>", file=sys.stderr)
        exit(-1)
    conf = SparkConf().setAppName('Windowed Streaming').setMaster("local[3]")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)

    ssc.checkpoint("file:///mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/socket/stateful")

    def update_func(new_value, last_num):
        return sum(new_value) + (last_num or 0)
    
    initialState = sc.parallelize([('hello', 1), ('world', 1)])
    lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    running_counts = lines.flatMap(lambda line: line.split(' '))\
                  .map(lambda word: (word, 1))\
                  .updateStateByKey(update_func, 
                                    initialRDD=initialState)
    running_counts.pprint()

    ssc.start()
    ssc.awaitTermination()
