"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: JumpGame.py
@time: 2019/2/23 21:30
"""

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # nums.reverse()
        m = 0
        for i, n in enumerate(nums):
            if i > m:
                return False
            m = max(m, i + n)
        return True
if __name__ == '__main__':
    test_list = [3,2,1,0,4]
    test_list = [3,2,1,0,4]
    print(Solution().canJump(test_list))