"""

栈 加 while

python 中使用 list 作为栈
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

    while(len(stack) >0):
        node = stack.pop();
        nibers = graph[node]
        for nd in nibers:
            if nd not  in seek:
                stack.append(nd)
                seek.append(nd)
        print(node)
        result.append(node)
    print(result)
dFS(G,"A")