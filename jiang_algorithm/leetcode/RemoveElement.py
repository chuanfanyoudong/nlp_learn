"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: RemoveElement.py
@time: 2019/2/19 22:22
"""
class Solution(object):
    def removeElement(self, nums, val):
        start, end = 0, len(nums) - 1
        while start <= end:
            if nums[start] == val:
                nums[start], nums[end], end = nums[end], nums[start], end - 1
            else:
                start += 1
        # return start
        print(nums)
        print(start)

        return start


if __name__ == '__main__':
    a = Solution()
    nums = [3,2,1,3]
    val = 3
    a.removeElement(nums, val)