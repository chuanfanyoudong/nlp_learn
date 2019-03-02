"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: MaximumSubarray.py
@time: 2019/2/23 20:56
"""


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 分别记录nums的第i个点的最大值，==前一个点的最大值与其和或者其本身
        max_sum, cure_sum = nums[0], nums[0]
        for i in range(len(nums[1:])):
            cure_sum = max(cure_sum + nums[i + 1], nums[i + 1])
            max_sum = max(cure_sum, max_sum)
        return max_sum

if __name__ == '__main__':
    test_list = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(Solution().maxSubArray(test_list))