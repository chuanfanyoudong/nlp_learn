#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: BubbleSort.py
@time: 2019/3/24 13:36
"""


def BubbleSort(myList):
    # 冒泡排序比较简单,就是对列表从头开始遍历，然后遍历他后面的所有之，如果比一开始的小，就交换这俩
    # 看下面的例子
    end = len(myList)
    # end 为 8
    for i in range(end):
        # 一开始i==0，
        for j in range(i,end):
            # 然后一次比较后面的所有之，看到了38，所以这俩交换，后面依次类推
            if myList[j] < myList[i]:
                myList[i], myList[j] = myList[j], myList[i]
    return myList

myList = [49,38,65,97,76,13,27,49]
print("Bubble Sort: ")
BubbleSort(myList)
print(myList)