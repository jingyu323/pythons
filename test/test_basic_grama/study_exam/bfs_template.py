"""
队列加 while
bfs全称是广度优先搜索，任选一个点作为起始点，然后选择和其直接相连的（按顺序展开）走下去。主要用队列实现，直接上图。两个搜索算法都只需要把图全都遍历下来就好

使用数组当作队列

"""
graph={
    "A":["B","C"],
    "B":["A","C","D"],
    "C":["A","B","D","E"],
    "D":["B","C","E","F"],
    "E":["C","D"],
    "F":["D"]
}

def BFS(graph,s):
    queue = []
    queue.append(s)
    seen = [] #保存访问过的节点
    seen.append(s)

    while (len(queue) > 0):
        vertex = queue.pop(0) # 弹出第一位 先进先出
        nodes=graph[vertex]
        for node in nodes:
            if node not  in seen:
                queue.append(node)
                seen.append(node)
        print(vertex)
BFS(graph,"A")