"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: ImplementstrStr.py
@time: 2019/2/19 22:40
"""


class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        for i in range(len(haystack) - len(needle)):
            if haystacck[i:i + len(needle)] == needle:
                return i
        return -1
