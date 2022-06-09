from itertools import permutations
from functools  import  cmp_to_key
arr = [1, 2, 3, 4]
for i in permutations([1, 2, 3, 4], 3):
    print(''.join(map(str, i)))

arr.sort(key=cmp_to_key())