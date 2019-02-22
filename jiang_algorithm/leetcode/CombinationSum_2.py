"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: CombinationSum_2.py
@time: 2019/2/21 22:57
"""
"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: CombinationSum.py
@time: 2019/2/21 22:08
"""


class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        # print(candidates)
        if sum([target >= i for i in candidates]) == 0:
            return []
            # print(result)
        result = []
        for i in range(len(candidates)):
            if candidates[i] == target:
                result.append([candidates[i]])
            pre = [j for j in self.combinationSum2(candidates[i+1:], target - candidates[i]) if j != []]
            if pre != []:
                for single in pre:
                    single.append(candidates[i])
                    result.append(single)
        final_result = []
        for i in result:
            i = sorted(i)
            if i not in final_result:
                final_result.append(i)
        # result = [final_result.append(j) for j in result if j != [] and j not in final_result]
        return final_result
if __name__ == '__main__':
    can = [10,1,2,7,6,1,5]
    tar = 8
    # print(set([[1,2,3], [1,2,3]]))
    print(Solution().combinationSum2(can, tar))
