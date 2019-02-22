"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: CombinationSum.py
@time: 2019/2/21 22:08
"""


class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        if sum([target >= i for i in candidates]) == 0:
            return []
            # print(result)
        result = []
        for i in range(len(candidates)):
            if candidates[i] == target:
                result.append([candidates[i]])
            pre = [j for j in self.combinationSum(candidates[i:], target - candidates[i]) if j != []]
            if pre != []:
                for single in pre:
                    single.append(candidates[i])
                    result.append(single)
        result = [j for j in result if j != []]
        return result
if __name__ == '__main__':
    can = [2,3,5]
    tar = 8
    print(Solution().combinationSum(can, tar))
