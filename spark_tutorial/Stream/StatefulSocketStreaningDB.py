'''
Author: ulysses
Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-03 13:17:11
LastEditors: Please set LastEditors
Description: 
'''
import sys

import pymysql
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

HOST = 'localhost'
user = 'root'
passwd = '62300313.a'

def save(rdd):
    print("========save===========")
    # 重新设置分区
    repartitionedRDD = rdd.repartition(3)
    # 将每个分区内数据保存到数据库
    repartitionedRDD.foreachPartition(save_to_db)

def save_to_db(records):
    # records 分区内的 k, v
    conn = pymysql.connect(host=HOST, port=3306, user=user,
                           passwd=passwd, db='spark', charset='utf8')
    cursor = conn.cursor()

    def do_insert(p):
        # sql = "insert into wordcount(word, count) values(%(word)s, %(count)s)"
        # value = {
        #     'word':p[0],
        #     'count': p[1]
        # }
        sql = "insert into wordcount(word, count) values(%s, %s)"
        try:
            cursor.execute(sql, (p[0], p[1]))
            conn.commit()
        except Exception as e:
            print(e)
            conn.rollback()
    for item in records:
        do_insert(item)
    

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
    # DStream是RDD集合 , 每个RDD 进行保存     
    running_counts.foreachRDD(save)

    ssc.start()
    ssc.awaitTermination()
