"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: DividedTwoIntegers.py
@time: 2019/2/20 22:49
"""

import math
class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        positive = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                i <<= 1 # i 记录的就是当前temp是几个divisor，是几个就加上几，这样就不是一个一个diviso的加了，是根据2的平方数加，所以会快。
                temp <<= 1
        if not positive:
            res = -res
        return min(max(-2147483648, res), 2147483647)

if __name__ == '__main__':
    a = Solution()
    print(a.divide(15,3))