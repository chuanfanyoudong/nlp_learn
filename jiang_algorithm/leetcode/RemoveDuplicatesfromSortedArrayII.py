"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: RemoveDuplicatesfromSortedArrayII.py
@time: 2019/3/3 13:43
"""


class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        new_result = []
        count_dict = {}
        i = 0
        while i < len(nums):
            target = nums[i]
            if target in count_dict:
                count_dict[target] += 1
                if count_dict[target] > 2:
                    nums.remove(target)
                else:
                    i += 1
            else:
                count_dict[target] = 1
                i += 1
        return nums



if __name__ == '__main__':
    print(Solution().removeDuplicates([0,0,1,1,1,1,2,3,3]))