"""
  某学校举行运动会,学生们按编号（1、2、3.....n)进行标识,
   现需要按照身高由低到高排列，
   对身高相同的人，按体重由轻到重排列，
   对于身高体重都相同的人，维持原有的编号顺序关系。
   请输出排列后的学生编号
   输入描述：
      两个序列，每个序列由N个正整数组成，(0<n<=100)。
      第一个序列中的数值代表身高，第二个序列中的数值代表体重，
   输出描述：
      排列结果，每个数据都是原始序列中的学生编号，编号从1开始，
   实例一：
      输入:
       4
       100 100 120 130
       40 30 60 50
      输出:
       2134
"""


class Stu:
    id = 0
    h = 0
    w = 0

    def __init__(self, id, h, w):
        self.id = id
        self.h = h
        self.w = w

    def __lt__(self, other):
        if self.h == other.h:
            return self.w - other.w
        else:
            return self.h - other.h


def exam():
    print("start")
    n = input()
    n = int(n)
    print(n)

    height = input()
    width = input()

    height_arr = height.split()
    width_arr = width.split()
    student_arr = []
    for i in range(n):
        print(i)
        student = Stu(i + 1, int(height[i],  int(width_arr[i])));
        student_arr.append(student)

    student_arr.sort()
    res = ""
    for st in student_arr:
        res = st.id + " "

    if len(res) > 1:
        substring = res[0:len(res) - 1]
        print(substring)


if __name__ == '__main__':
    exam()
