from queue import Queue


class MyStack():
    def __init__(self):
        self.data = Queue()

    def push(self, value):
        tmp_queue = Queue()
        tmp_queue.put(value)
        while not self.data.empty():
            tmp_queue.put(self.data.get())
        while not tmp_queue.empty():
            self.data.put(tmp_queue.get())

    def pop(self):
        return self.data.get()

    def top(self):
        value = self.data.get()
        self.push(value)
        return value

    def empty(self):
        return self.data.empty()

    def  isvalid(self,s):
        d = {"(": ")", "[": "]", "{": "}"}
        stack = []

        for char in s:
            if char in d:
                stack.append(char)
            else:

                print(stack[-1],"  ==== ===  ")
                if stack and d[stack[-1]] == char :
                    stack.pop()
                else:
                    return False
        if stack:
            return False
        return True



if __name__ == '__main__':
    stack = MyStack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    stack.push(4)
    stack.push(5)

    print(stack.top())

    print("============================")

    while not stack.empty():
        print(stack.pop())

    res = stack.isvalid("()[]{}")
    print(res)
