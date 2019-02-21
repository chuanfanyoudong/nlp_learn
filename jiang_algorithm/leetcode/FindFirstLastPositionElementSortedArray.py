"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: FindFirstLastPositionElementSortedArray.py
@time: 2019/2/21 20:22
"""


class Solution(object):
    def searchRange(self, nums, target):
        """
        注意边界
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(nums) == 0:
            return [-1,-1]
        if len(nums) == 1:
            if target == nums[0]:
                return [0,0]
            else:
                return [-1,-1]
        low, high = 0, len(nums)-1
        while low <= high:
            mid = int((low + high)//2)
            if nums[mid] == target:
                left = right = mid
                while left - 1 >= 0 and nums[left - 1] == target:
                    left = left - 1
                while right + 1 <= len(nums) - 1 and nums[right + 1] == target:
                    right += 1
                return [left, right]
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return [-1, -1]