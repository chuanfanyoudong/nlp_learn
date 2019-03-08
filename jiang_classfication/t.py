
#!/usr/bin/env python
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com/chuanfanyoudong
@software: PyCharm
@file: t.py
@time: 2019/3/8 16:19
"""

from sklearn.metrics import f1_score
y_true = ["1","1","1"]
y_pred = ["1","1","1"]
f1 = f1_score(y_true, y_pred, average='macro')
print(f1)