"""
@author: zkjiang
@contact: jiang_zhenkang@163.com
@software: PyCharm
@file: GroupAnagrams.py
@time: 2019/2/22 21:47
"""


class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        record_dict = {}
        for i in strs:
            sort_i = "".join(sorted(list(i)))
            if sort_i not in record_dict:
                record_dict[sort_i] = [i]
            else:
                record_dict[sort_i].append(i)
        return list(record_dict.values())

if __name__ == '__main__':
    print(list("123"))
    print(Solution().groupAnagrams(   ["eat", "tea", "tan", "ate", "nat", "bat"]))