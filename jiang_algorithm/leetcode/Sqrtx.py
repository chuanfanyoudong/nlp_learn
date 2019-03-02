"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Sqrtx.py
@time: 2019/2/26 22:39
"""
test = 8

# class Solution(object):
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        low = 0
        high = x
        result = 0
        while low <= high:
            mid = (low + high) // 2
            if mid * mid <= x:
                low = mid + 1
                result = mid  # 每次记下平方小于8的那个值，防止以后没有平方小于8的值了
            else:
                high = mid - 1
        return int(result)


if __name__ == '__main__':
    print(Solution().mySqrt(8))