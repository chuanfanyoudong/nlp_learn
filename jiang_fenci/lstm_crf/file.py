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
import torch as t
from torch.autograd import Variable as v
m = v(t.FloatTensor([[2, 3]]), requires_grad=True)
# j = t.zeros(2 ,2)
# k = v(t.zeros(1, 2))
# m.grad.data.zero_()
# k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
# k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
# simple gradient
j = t.zeros(2 ,2)
k = v(t.zeros(1, 2))
# m.grad.data.zero_()
k[0, 0] = m[0, 0] ** 2 + 3 * m[0 ,1]
k[0, 1] = m[0, 1] ** 2 + 2 * m[0, 0]
k.backward(t.FloatTensor([[1, 0]]), retain_variables=True)
j[:, 0] = m.grad.data
m.grad.data.zero_()
k.backward(t.FloatTensor([[0, 1]]))
j[:, 1] = m.grad.data
print('jacobian matrix is')
print(j)