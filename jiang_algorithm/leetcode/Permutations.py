"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Permutations.py
@time: 2019/2/22 20:58
"""


class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        for i in range(len(nums)):
            pre = self.permute(nums[:i] + nums[i+1:])
            result += [[nums[i]] + single for single in pre]
        return result
if __name__ == '__main__':
    print(Solution().permute( [1,2,3]))