class Foo: #class是关键字，Foo是类名
  class_int = 10 #类变量
  #创建类中的函数(方法)
  def bar(self,name):   #self是特殊参数，类的实例化，必须填写。
    print('bar',name)

obj = Foo() #根据Foo创建对象obj
print('类访问类变量：',Foo.class_int)
print('对象访问类变量：', obj.class_int)
obj.bar(3) 