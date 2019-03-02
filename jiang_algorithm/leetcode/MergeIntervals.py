"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: MergeIntervals.py
@time: 2019/2/23 21:46
"""


# Definition for an interval.
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        out = []
        for i in sorted(intervals, key=lambda i: i.start):
            if out and i.start <= out[-1].end:
                out[-1].end = max(i.end. out[-1].end)
            else:
                out.append(i)