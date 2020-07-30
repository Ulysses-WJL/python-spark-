'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-17 10:02:48
@LastEditors: your name
@Description: 
@可以输入预定的版权声明、个性签名、空行等
'''
# %%
from pyspark import SparkContext

import findspark
findspark.init()

# sc = SparkContext('hadoop-master:7077', 'test')
from pyspark import SparkConf
from pyspark.sql import SparkSession
import traceback
 
appname = "test"#任务名称
master ="local"#单机模式设置
'''
local: 所有计算都运行在一个线程当中，没有任何并行计算，通常我们在本机执行一些测试代码，或者练手，就用这种模式。
local[K]: 指定使用几个线程来运行计算，比如local[4]就是运行4个worker线程。通常我们的cpu有几个core，就指定几个线程，最大化利用cpu的计算能力
local[*]: 这种模式直接帮你按照cpu最多cores来设置线程数了。
'''

conf = SparkConf().setAppName(appname).setMaster(master)#spark资源配置
spark=SparkSession.builder.config(conf=conf).getOrCreate()
sc=spark.sparkContext
try:
    words = sc.parallelize(
        ["scala",
         "java",
         "hadoop",
         "spark",
         "akka",
         "spark vs hadoop",
         "pyspark",
         "pyspark and spark"
         ])
    counts = words.count()
    print("Number of elements in RDD is %i" % counts)
    sc.stop()
    print('计算成功！')
except:
    sc.stop()
    traceback.print_exc()#返回出错信息
    print('连接出错！')
# %%
