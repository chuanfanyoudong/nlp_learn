#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: quicksort.py
@time: 2019/2/27 9:06
"""

<<<<<<< Updated upstream
# 最坏的情况下，快排并不比插入排序号，也是n2

=======
>>>>>>> Stashed changes
def QuickSort(myList,start,end):
    #判断low是否小于high,如果为false,直接返回
    if start < end:
        i,j = start,end
        #设置基准数
        base = myList[i]

        while i < j:
<<<<<<< Updated upstream
            # 核心逻辑就是碰到小的，就换到前面去，碰到大的就换到后面去，最后在把base换给list【i】
            #如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while (i < j) and (myList[j] >= base):
                j = j - 1
            #如找到,则把第j个元素赋值给第个元素i,此时表中i,j个元素相等
            myList[i] = myList[j]
=======
            #如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while (i < j) and (myList[j] >= base):
                j = j - 1

            #如找到,则把第j个元素赋值给第个元素i,此时表中i,j个元素相等
            myList[i] = myList[j]

>>>>>>> Stashed changes
            #同样的方式比较前半区
            while (i < j) and (myList[i] <= base):
                i = i + 1
            myList[j] = myList[i]
        #做完第一轮比较之后,列表被分成了两个半区,并且i=j,需要将这个数设置回base
        myList[i] = base

        #递归前后半区
        QuickSort(myList, start, i - 1)
<<<<<<< Updated upstream
        QuickSort(myList, i + 1, end)
=======
        QuickSort(myList, j + 1, end)
>>>>>>> Stashed changes
    return myList


myList = [49,38,65,97,76,13,27,49]
print("Quick Sort: ")
QuickSort(myList,0,len(myList)-1)
print(myList)