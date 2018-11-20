list1 = ['S', 'B', 'M', 'E']
lis2 = [1, 3, 1, 3, 0, 0, 1, 3, 1, 3, 0, 1, 3]


map = list(map(lambda x:list1[x], lis2))
for i in map:
    print(i)
print(list(map))

