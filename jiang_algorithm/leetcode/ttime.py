"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: ttime.py
@time: 2019/2/19 22:02
"""

import time
start = time.time()

i = 0
while i < 10000:
    a,b,c = 3,4,5
    i += 1

end = time.time()
print(end - start)
