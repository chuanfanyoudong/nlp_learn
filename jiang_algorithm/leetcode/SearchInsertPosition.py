"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: SearchInsertPosition.py
@time: 2019/2/21 20:34
"""

def searchInsert(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if len(nums) == 0:
        return 0
    if nums[0] >= target:
        return 0
    if nums[-1] <= target:
        return len(nums) - 1
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            high = mid - 1

        if nums[mid] < target:
            low = mid + 1
    if nums[mid] < target:
        return mid + 1
    else:
        return mid


if __name__ == '__main__':
    nums = [1,3, 5,6]
    target = 7
    print(Solution().searchInsert(nums, target))