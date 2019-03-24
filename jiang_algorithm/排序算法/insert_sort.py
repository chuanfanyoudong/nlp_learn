#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: insert_sort.py
@time: 2019/3/24 13:20
"""

def InsertSort(myList):
    #判断low是否小于high,如果为false,直接返回
    end = len(myList)
    for i in range(1, end):
        # 核心逻辑就是把mylist[i]插入到mylist[:i-1]中
        # 所以先定好边界，i - 1
        j = i-1
        # 去除mulist[i]
        base = myList[i]
        # 和mylist[:i-1]中的每一个作比较
        while j >= 0:
            # 如果base比mylist[j]小则把base查到j前面，也就是把mylist[j]赋值给mylist[j+1],把base赋值给mylist[j]
            if base < myList[j]:
                myList[j + 1] = myList[j]
                myList[j] = base
            j -= 1
    return myList


myList = [49,38,65,97,76,13,27,49]
print("Insert Sort: ")
InsertSort(myList)
print(myList)