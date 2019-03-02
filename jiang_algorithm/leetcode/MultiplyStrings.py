"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: MultiplyStrings.py
@time: 2019/2/22 20:21
"""


class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        if num2 == "0" or num1 == "0":
            return "0"
        result = 0
        for i in range(len(num1)):
            nums_0 = len(num1)- 1 - i
            token = num1[i]
            result += int(str(int(token) * int(num2)) + "0"*nums_0)
        return str(result)

if __name__ == '__main__':
    print(Solution().multiply("12", "12"))


