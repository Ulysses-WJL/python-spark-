'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
LastEditTime: 2020-08-17 16:05:05
LastEditors: ulysses
@Description: 
'''
from pyspark import SparkConf, SparkContext
from operator import gt

class SecondarySort:
    def __init__(self, k):
        self.c1 = k[0]
        self.c2 = k[1]
    def __gt__(self, other):
        if other.c1 == self.c1:
            return gt(self.c2, other.c2)
        else:
            return gt(self.c1, other.c1)


def main():
    conf = SparkConf().setAppName('spark_sort').setMaster('local[1]')
    sc = SparkContext(conf=conf)

    rdd_04 = sc.textFile("../data/file4.txt")
    res1 = rdd_04.filter(lambda line: len(line.strip()) > 0)

    res2 = res1.map(
        lambda x: ((int(x.split(" ")[0]), int(x.split(" ")[1])),
                x)
    )
    res2.collect()
    res3 = res2.map(lambda x: (SecondarySort(x[0]), x[1]))
    res4 = res3.sortByKey(False)
    res5 = res4.map(lambda x: x[1])
    res5.foreach(print)


if __name__ == "__main__":
    main()