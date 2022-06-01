N =int(input())
input_res=set()
for i in  range(N):
    input_res.add(int(input()))

lee =sorted(input_res)
for i in lee:
    print(i)