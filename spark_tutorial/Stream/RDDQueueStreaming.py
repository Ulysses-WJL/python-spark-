'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-08-02 20:26:33
@LastEditors: Please set LastEditors
@Description: 
'''
import time
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext



if __name__ == "__main__":
    sc = SparkContext(master='local[2]', appName="PythonStreamingQueueStream")
    ssc = StreamingContext(sc, 2)
    # 创建一个队列，通过该队列可以把RDD推给一个RDD队列流
    rdd_queue = []
    for i in range(5):
        rdd_queue.append(ssc.sparkContext.parallelize(
            [j for j in range(1, 1001)],
            10
        ))
        time.sleep(1)
    # 创建一个RDD队列流
    input_stream = ssc.queueStream(rdd_queue)
    mapped_stream = input_stream.map(lambda x: (x % 10, 1))
    reduced_stream = mapped_stream.reduceByKey(lambda x, y: x + y)
    reduced_stream.pprint()

    ssc.start()
    ssc.stop(stopSparkContext=True, stopGraceFully=True)
