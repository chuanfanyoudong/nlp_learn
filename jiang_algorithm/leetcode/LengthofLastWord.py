"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: LengthofLastWord.py
@time: 2019/2/23 21:50
"""


class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
            return 0
        if " " not in s:
            return len(s)
        return len(s.strip().split(" ")[-1])



if __name__ == '__main__':
    test_str = "Hello World"
    print(Solution().lengthOfLastWord(test_str))