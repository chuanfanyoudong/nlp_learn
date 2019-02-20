"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: removeDuplicates.py
@time: 2019/2/19 22:14
"""

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        tag = 0
        for i in range(1,len(nums)):
            if nums[i] != nums[tag]:
                tag += 1
                nums[tag] = nums[i]
        return tag + 1