"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Powxn.py
@time: 2019/2/22 22:02
"""


class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            return 1/x**n
        if n > 0:
            return x**n


print(2%2)