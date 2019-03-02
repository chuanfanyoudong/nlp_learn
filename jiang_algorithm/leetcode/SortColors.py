"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SortColors.py
@time: 2019/2/27 22:47
"""
test_list = [2,0,2,1,1,0]

class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        i = 0
        start = 0
        end = len(nums) - 1
        while i < end:
            if nums[i] == 0:
                nums[start], nums[i] = nums[i], nums[start]
                start += 1
            if nums[i] == 2:
                nums[end], nums[i] = nums[i], nums[end]
                end -= 1
                i -= 1
            i += 1
        print(nums)


if __name__ == '__main__':
    print(Solution().sortColors(test_list))



