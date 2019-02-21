"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: CountSay.py
@time: 2019/2/21 21:50
"""
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        s = "1"
        if n == 1:
            return "1"
        str_result = self.countAndSay(n - 1) + "*"
        result = ""
        count = 1
        for i in range(len(str_result)):
            if str_result[i] == "*":
                return result
            if str_result[i] == str_result[i + 1]:
                count += 1
            else:
                result += str(count) + str_result[i]
                count = 1

if __name__ == '__main__':
    print(Solution().countAndSay(5))