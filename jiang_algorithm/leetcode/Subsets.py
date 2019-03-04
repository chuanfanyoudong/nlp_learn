"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: Subsets.py
@time: 2019/3/2 19:41
"""

test_list = [1,2,3]

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) == 1:
            return [[],nums]
        return [[nums[0]] + i for i in self.subsets(nums[1:])] +  self.subsets(nums[1:])



if __name__ == '__main__':
    print(Solution().subsets(test_list))