from itertools import permutations
from functools  import  cmp_to_key
arr = [1, 2, 3, 4]
for i in permutations([1, 2, 3, 4], 3):
    print(''.join(map(str, i)))

def com(el,e2):

    return  el - e2

arr.sort(key=cmp_to_key())