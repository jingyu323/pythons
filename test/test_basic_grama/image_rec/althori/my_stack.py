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
