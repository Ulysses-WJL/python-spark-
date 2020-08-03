'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 11:31:11
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
    ssc = StreamingContext(sc, 10)
    # 防止数据丢失
    ssc.checkpoint("file:///mnt/data1/workspace/data_analysis_mining/Python_Spark/spark_tutorial/data/socket/checkpoint")

    lines = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))
    counts = lines.flatMap(lambda line: line.split(' '))\
                  .map(lambda word: (word, 1))\
                  .reduceByKeyAndWindow(
                      lambda x, y: x+y,
                      lambda x, y: x-y,
                      30, 10
                  )
    # 对进入滑动窗口的新数据进行reduce操作，
    # 并对离开窗口的老数据进行“逆向reduce”操作
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()