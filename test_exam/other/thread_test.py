import threading
import time
from queue import Queue


def thread_job():
    print('This is a thread of %s' % threading.current_thread())


def main():
    thread = threading.Thread(target=thread_job, )  # 定义线程
    # thread.daemon = False
    thread.start()  # 让线程开始工作


def job1():
    print("start the  1st threading")
    time.sleep(0.5)
    print("1st threading ends")
def job2():
    print("start the  2nd threading")
    time.sleep(2)
    print("2nd threading ends")
def job3():
    print("start the  3rd threading")
    time.sleep(1)
    print("3rd threading ends")


def job(l,q):
    for i in range (len(l)):
        l[i] = l[i]**2
    q.put(l)

def multithreading():
    q =Queue()
    threads = []
    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]
    for i in range(4):
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)


def job5():
    global A, lock
    lock.acquire()
    for i in range(2):
        print("get A_value of job1",A)
        A+=1
        print("A + 1 = %d"%(A))
        time.sleep(1)
    print("get A_value of job1",A)
    lock.release()
def job6():

    global A, lock
    lock.acquire()
    for i in range(2):
        print("get A_value of job2",A)
        A+=10
        print("A + 10 = %d"%(A))
        time.sleep(3)
    print("get A_value of job2",A)
    lock.release()




if __name__ == '__main__':
    print("1")
    main()
    print("2")

    print("begins")
    lock = threading.Lock()
    first = threading.Thread(target=job1)
    second = threading.Thread(target=job2)
    third = threading.Thread(target=job3)
    # join方法存按代码至上而下先后执行(必须得执行完中间代码的子线程，才会去执行末行主线程代码)---阻塞式
    first.start()
    first.join()
    second.start()
    second.join()
    third.start()
    third.join()
    print("ends")

    multithreading()

    print("begin")
    lock1 = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job6)
    t2 = threading.Thread(target=job5)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("end")
