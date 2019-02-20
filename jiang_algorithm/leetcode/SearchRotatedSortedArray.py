"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SearchRotatedSortedArray.py
@time: 2019/2/20 23:28
"""

class Solution():
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        low, high = 0, len(nums)-1
        while low <= high:
            mid = int((low + high)/2)
            if nums[mid] == target:
                return mid
            if nums[mid] >= nums[low]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
        return -1
        # for i in range(len(nums)):
        #     if nums[i] == target:
        #         return i


if __name__ == '__main__':
    a = Solution()
    res = a.search([5,6,7,1,2,3,4],4)
    print(res)
