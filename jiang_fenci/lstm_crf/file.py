#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: file.py
@time: 2019/1/10 14:49
"""
import math

import numpy as np
from sklearn_crfsuite import metrics

a = np.array([[9,2,0],[4,0,0],[5,6,7]])
# np.sort(a, axis= 0, kind= "quicksort", order= [3, 5, 1])
order= [2, 1, 3]
# print(a)
all_tags = [[1,2,3], [1,2]]
all_pre_tags = [[1,2,2], [1,2]]
print(metrics.flat_classification_report(all_tags, all_pre_tags, labels=[0, 1, 2, 3], digits=3))
F1 = metrics.flat_f1_score(all_tags, all_pre_tags, average='weighted', labels=[0, 1, 2, 3])
print("F1:", F1)