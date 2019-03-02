"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Permutations2.py
@time: 2019/2/22 21:11
"""


class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = []
        if len(nums) == 0:
            return []
        if len(nums) == 1:
            return [nums]
        single_list = []
        for i in range(len(nums)):
            if nums[i] not in single_list:
                single_list.append(nums[i])
                pre = self.permuteUnique(nums[:i] + nums[i+1:])
                result += [[nums[i]] + single for single in pre]
        return result
if __name__ == '__main__':
    print(Solution().permuteUnique( [1,1,3]))