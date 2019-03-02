"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: PlusOne.py
@time: 2019/2/26 21:47
"""

test_list = [0]

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        idx = len(digits)-1
        num = digits[idx]
        while idx > 0 and num == 9:
            digits[idx] = 0
            idx = idx - 1
            num = digits[idx]
        if digits[idx] == 9:
            digits = [1,0] + digits[1:]
        else:
            digits[idx] += 1
        return digits


if __name__ == '__main__':
    print(Solution().plusOne(test_list))