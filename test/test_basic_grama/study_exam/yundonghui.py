from functools import cmp_to_key
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
       
       
       public class Main93 {

  /*
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
   */
  static class Stu {
    int id;
    int h;
    int w;

    public Stu(int id, int h, int w) {
      this.id = id;
      this.h = h;
      this.w = w;
    }
  }

  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    int n = Integer.parseInt(in.nextLine());
    String[] h = in.nextLine().split(" ");
    String[] w = in.nextLine().split(" ");
    in.close();
    LinkedList<Stu> stus = new LinkedList<>();
    for (int i = 0; i < n; i++) {
      Stu stu = new Stu(i + 1, Integer.parseInt(h[i]), Integer.parseInt(w[i]));
      stus.add(stu);
    }
    stus.sort((o1, o2) -> o1.h == o2.h ? (o1.w - o2.w) : o1.h - o2.h);
    StringBuilder builder = new StringBuilder();
    stus.forEach(x -> builder.append(x.id).append(" "));
    System.out.println(builder.substring(0, builder.length() - 1));
  }
"""


class Stu:
    # id = 0
    # h = 0
    # w = 04

    def __init__(self, id, h, w):
        self.id = id
        self.h = h
        self.w = w

    def __lt__(self, other):
        if self.h == other.h:
            print(self.w - other.w)
            return self.w - other.w >0
        else:
            return self.h - other.h >0

def cmp(kid1, kid2):


    if kid1.h == kid2.h:
        print(kid1.w - kid2.w , "----------------")
        return (kid1.w - kid2.w) < 0
    else:
        print(kid1.h - kid2.h , "------3----------")
        return (kid1.h - kid2.h) < 0


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
        print(height_arr[i] ,width_arr[i])
        student = Stu(i + 1, int(height_arr[i]), int(width_arr[i]));
        student_arr.append(student)

    # student_arr.sort()
    student_arr.sort(key=cmp_to_key(cmp))
    res = ""
    for st in student_arr:
        res = res+ str(st.id) + " "

    if len(res) > 1:
        substring = res[0:len(res) - 1]
        print(substring)
    student_arr.sort()
    res = ""
    for st in student_arr:
        res = res+ str(st.id) + " "
    print(res)
if __name__ == '__main__':
    exam()
