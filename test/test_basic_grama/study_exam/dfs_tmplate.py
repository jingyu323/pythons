"""

栈 加 while

python 中使用 list 作为栈

可以用 字典 来存储，key 为顶点，value 为相邻顶点的列表（如果考虑边的权值，则 value 为包含了边权重的字典）

注意区别：DFS采用的是栈的结构，体现在将BFS中的queue.pop（0）
(取出第一个元素，“先进先出”)改为栈stack.pop()默认是去除列表中的最后一个元素(”先进后出”）
"""

graph={
    "A":["B","C"],
    "B":["A","C","D"],
    "C":["A","B","D","E"],
    "D":["B","C","E","F"],
    "E":["C","D"],
    "F":["D"]
}

G = {
	'A': ['B', 'C', 'G'],
	'B': ['A', 'D', 'G', 'E'],
	'C': ['A'],
	'D': ['B', 'C', 'E'],
	'E': ['B', 'D', 'F'],
	'F': ['E'],
	'G': ['A', 'B']
}


def dFS(graph,s):
    result = []
    stack = []
    seek = []
    stack.append(s)
    seek.append(s)
    parent = {s: None}  # 字典

    while(len(stack) >0):
        node = stack.pop(); # 弹出最后一位
        nibers = graph[node]
        for nd in nibers:
            if nd not  in seek:
                stack.append(nd)
                seek.append(nd)
        print(node)
        result.append(node)
    print(result)
dFS(graph,"A")