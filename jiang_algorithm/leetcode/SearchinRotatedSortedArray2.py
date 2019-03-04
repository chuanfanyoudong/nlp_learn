"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SearchinRotatedSortedArray2.py
@time: 2019/3/3 14:03
"""

## 注意边界，边界很烦躁

test_list = [2,5,6,0,0,1,2]
test_list = [1,3,5]

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high)//2
            if nums[mid] == target:
                return True
            while low  < mid and nums[low] == nums[mid]:
                low += 1
            if nums[low] <= nums[mid]:
                if nums[low] <= target < nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] < target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
        return False

print(Solution().search(
    test_list, 1
))



