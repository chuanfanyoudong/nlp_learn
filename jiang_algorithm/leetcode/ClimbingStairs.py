"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: ClimbingStairs.py
@time: 2019/2/26 22:56
"""


class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        result = [0,1,2]
        if n == 1:
            return 1
        elif n == 2:
            return 2
        for i in range(3,n +1):
            result.append(result[i-1] + result[i-2])
        return result[n]

if __name__ == '__main__':
    print(Solution().climbStairs(3))