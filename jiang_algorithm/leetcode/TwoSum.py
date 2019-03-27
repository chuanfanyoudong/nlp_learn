"""
leetcode-1
在一个list中求出两个数之和为target值的两个数的坐标
"""


class Solution(object):
    def twoSum(self, nums, target):
        """
        可以用暴力方法，不太好
        可以用见一个字典，字典的key为list的每一个值，然后遍历一遍，同时看一下list的值有没有在字典中，因为判断在不在字典中的复杂度为O（1）
        """
        # 一定要先判断特殊情况
        if len(nums) < 2:
            return None
        # 创建一个临时字典
        tmp_dict = {}
        # 遍历nums中的每一个idx
        for i in range(len(nums)):
            # target与当前值的差为diff，如果diff也在字典中说明，存在这样的对，返回两个左边就可以
            diff = target - nums[i]
            # 如果diff在字典中，则返回两个坐标
            if diff in tmp_dict:
                return [tmp_dict[diff], i]
            # 如果不在字典中，则将当前值作为key，当前值的坐标作为value放入字典
            else:
                tmp_dict[nums[i]] = i
        return None

