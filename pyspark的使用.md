<!--
 * @Author: ulysses
 * @Date: 1970-01-01 08:00:00
 * @LastEditTime: 2020-08-01 16:12:30
 * @LastEditors: Please set LastEditors
 * @Description: 
 * @可以输入预定的版权声明、个性签名、空行等
--> 
# pyspark

## 1. 本地运行

```
pyspark --master local[4]
```
`local[N]`代表在本地运行, 使用N个线程, `local[*]`会尽量使用机器上的CPU核心.
```
>>> sc.master
'local[4]'
```

**读取本地文件**
计算行数
```
>>> text_file = sc.textFile("file:/root/input/LICENSE.txt")
>>> text_file.count()
289
```

**读取HDFS文件**
```
>>> text_file = sc.textFile("hdfs://hadoop-master:9000/input_fruit/fruit.tsv")
>>> text_file.count()
3
```

## 2. YARN 运行pyspark

设置环境变量
```
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
```
让YARN 帮助Spark进行资源的管理
```
pyspark --master yarn --deploy-mode client
```
查看当前运行模式
```
>>> sc.master
'yarn'
```

重要参数说明
```
  --executor-memory 1G	指定每个executor可用内存为1G
  --total-executor-cores 2	指定所有executor使用的cpu核数为2个
  --executor-cores	指定每个executor使用的cpu核数
  --master MASTER_URL         spark://host:port, mesos://host:port, yarn,
                              k8s://https://host:port, or local (Default: local[*]).
  --deploy-mode DEPLOY_MODE   Whether to launch the driver program locally ("client") or
                              on one of the worker machines inside the cluster ("cluster")
                              (Default: client).

```

## 3. Spark Standalone Cluster

只使用Spark自身节点运行的集群模式，也就是我们所谓的独立部署（Standalone）模式。Spark的Standalone模式体现了经典的master-slave模式。

设置`spark-env.sh`
```
export JAVA_HOME=/usr/local/java
export SPARK_MASTER_HOST=hadoop-master
export SPARK_WORKER_CORES=2
export SPARK_WORKER_MEMORY=512m
```
配置`slaves`
```
hadoop-master
hadoop-slave1
hadoop-slave2
```

```
pyspark --master spark://hadoop-master:7077 \
--total-executor-cores 3 \
--num-executors 1 \
--executor-memory 512m 
```

```
>>> sc.master
'spark://hadoop-master:7077'

```

## 配置历史服务

修改spark-default.conf文件，配置日志存储路径
```
spark.eventLog.enabled          true
spark.eventLog.dir               hdfs://hadoop-master:9000/spark_log
```

修改spark-env.sh文件, 添加日志配置

```
export SPARK_HISTORY_OPTS="
-Dspark.history.ui.port=18080 
-Dspark.history.fs.logDirectory=hdfs://hadoop-master:9000/spark_log 
-Dspark.history.retainedApplications=30"
```

YARN模式的历史服务器, 修改spark-defaults.conf
```
spark.yarn.historyServer.address=linux1:18080
spark.history.ui.port=18080
```
启动历史服务器
```
start-history-server.sh 
```

##  spark 高可用
所谓的高可用是因为当前集群中的Master节点只有一个，所以会存在单点故障问题。所以为了解决单点故障问题，需要在集群中配置多个Master节点，一旦处于活动状态的Master发生故障时，由备用Master提供服务，保证作业可以继续执行。这里的高可用一般采用Zookeeper设置

3)修改spark-env.sh文件添加如下配置
注释如下内容：
```
#SPARK_MASTER_HOST=linux1
#SPARK_MASTER_PORT=7077
SPARK_MASTER_WEBUI_PORT=8989

export SPARK_DAEMON_JAVA_OPTS="
-Dspark.deploy.recoveryMode=ZOOKEEPER 
-Dspark.deploy.zookeeper.url=hadoop-master,hadoop-slave1,hadoop-slave2
-Dspark.deploy.zookeeper.dir=/spark"
```

提交应用到高可用集群
```
--master spark://linux1:7077,linux2:7077 \
--deploy-mode cluster \
```

## 配置IPython

安装anaconda, 修改bashrc, 
```
export PATH=/root/miniconda3/bin:$PATH
export PYSPARK_PYTHON=/root/miniconda3/bin/python
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip=0.0.0.0 --allow-root'
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$PYTHONPATH

```
创建目录
```
root@hadoop-master:~# mkdir -p pyspark_project
root@hadoop-master:~# cd pyspark_project/
root@hadoop-master:~/pyspark_project# pyspark

```

YARN模式
```
pyspark --master yarn-client
```
SPARK Standalone 模式

``` 
pyspark --master spark://hadoop-master:7077 \
--total-executor-cores 3 \
--num-executors 1 \
--executor-memory 512m 
```

# spark-submit 
使用 `spark-submit`执行Python Spark application 

|选项 |说明|
--|--
|--master MASTER_URL| 设置Spark 在什么环境运行|
|--driver-memory MEM| driver程序所使用的内存|
|--executor-memory MEM| executor 程序使用的内存|
|--name NAME| application的名称|
|python文件名|要运行的Python 程序|



--masster 选项设置
|MASTER_URL|说明|
--|--
|local|本地 单线程|
|local[k]|本地 K个线程|
|local[*]|本地 Spark 会自动尽量使用本地计算机上的多核|
|spark://HOST:PORT| Spark Standalone Cluster 如: spark://master:7077|
|YARN|在YARN Client上运行 必须要设置HADOOP_CONF_DIR 或 YARN_CONF_DIR|

例:
```
spark-submit --driver-memory 2g --master local[4] WordCount.py
```

# Spark 读写 HBase数据

把HBase的lib目录下的一些jar文件拷贝到Spark中，这些都是编程时需要引入的jar包，需要拷贝的jar文件包括：所有hbase开头的jar文件、guava-12.0.1.jar、htrace-core-3.1.0-incubating.jar和protobuf-java-2.5.0.jar'
```
$ cd /usr/local/spark/jars
$ mkdir hbase
$ cd hbase
$ cp /usr/local/hbase/lib/hbase*.jar ./
$ cp /usr/local/hbase/lib/guava-12.0.1.jar ./
$ cp /usr/local/hbase/lib/htrace-core-3.1.0-incubating.jar ./
$ cp /usr/local/hbase/lib/protobuf-java-2.5.0.jar ./
```

此外，在Spark 2.0以上版本中，缺少把HBase数据转换成Python可读取数据的jar包，需要另行下载。可以访问下面地址下载spark-examples_2.11-1.6.0-typesafe-001.jar：

https://mvnrepository.com/artifact/org.apache.spark/spark-examples_2.11/1.6.0-typesafe-001

下载以后保存到“/usr/local/spark/jars/hbase/”目录中

spark-env.sh文件 设置
```
export SPARK_DIST_CLASSPATH=$(/usr/local/hadoop/bin/hadoop classpath):$(/usr/local/hbase/bin/hbase classpath):/usr/local/spark/jars/hbase/*
```

读取hbase数据:
```
host = "hadoop-master"
table = "student"
conf = {"hbase.zookeeper.quorum": host, "hbase.mapreduce.inputtable": table}
keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"

hbase_rdd = sc.newAPIHadoopRDD("org.apache.hadoop.hbase.mapreduce.TableInputFormat",
                               "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
                               "org.apache.hadoop.hbase.client.Result",
                               keyConverter=keyConv,
                               valueConverter=valueConv,
                               conf=conf)
hbase_rdd.cache()      
output = hbase_rdd.collect()
for k, v in output:
    print(k, v) 
```
写入数据