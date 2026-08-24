'''
前提是你已经装好了 Python 和 Django

步骤1：

    django-admin startproject myblog  # 创建 myblog 项目
步骤2：
 cd myblog  并创建应用
   python manage.py startapp blog   # 创建blog应用
步骤3：
    打开settings.py 配置文件，添加blog应用

         INSTALLED_APPS -》 'blog',

步骤4：步骤配置数据库：

    DATABASES =》 修改相应配置  // https://docs.djangoproject.com/en/2.1/ref/settings/#databases

    Django默认帮我们做很多事情，比如User、Session 这些都需要创建表来存储数据，
    Django已经把这些模块帮我准备好了，我们只需要执行数据库同步，把相关表生成出来即可：
    python manage.py migrate   //执行之前必须先把数据库给创建了

步骤5：要想登录admin后台，必须要有帐号，接下来创建超级管理员帐号

python manage.py createsuperuser
启动应用
python manage.py runserver
登录后台地址
http://127.0.0.1:8000/admin/


步骤6：定义blog数据结构
blog目录下的models.py文件 定义数据模型
创建数据模型改变记录
python manage.py makemigrations blog
同步表结构
python manage.py migrate



执行之后报错 ：Did you install mysqlclient?

其实python3 基本不用装  mysqlclient 可能是2 的后遗症

pip install mysql-python   ------没有python3.0以上的版本   具体不知道

 先安装 pip install mysql-connector-python 即可
 再装 pip intalll mysqlclient


 如果不行可先装下mysql 在执行
 brew install mysql





    manage.py ： Django项目里面的工具，通过它可以调用django shell和数据库等。

    myblog/

    | ---  settings.py ： 包含了项目的默认设置，包括数据库信息，调试标志以及其他一些工作的变量。

    | ---  urls.py ： 负责把URL模式映射到应用程序。

    | --- wsgi.py :  用于项目部署。

    blog /

    | --- admin.py  :  django 自带admin后面管理，将models.py 中表映射到后台。

    | --- apps.py :  blog 应用的相关配置。

    | --- models.py  : Django 自带的ORM，用于设计数据库表。

    | --- tests.py  :  用于编写Django单元测试。

    | --- veiws.py ：视图文件，用于编写功能的主要处理逻辑。


CBV和FBV
我们之前写过的都是基于函数的view，就叫FBV。还可以把view写成基于类的  叫CBV。


=================


##1、安装方法   https://github.com/twz915/DjangoUeditor3/

* 方法一：将github整个源码包下载回家，在命令行运行：
	python setup.py install
* 方法二：使用pip工具在命令行运行(推荐)：
    pip install DjangoUeditor
##2、在Django中安装DjangoUeditor 在INSTALL_APPS里面增加DjangoUeditor app，如下： INSTALLED_APPS = ( #........ 'DjangoUeditor', ) ##3、配置urls url(r'^ueditor/',include('DjangoUeditor.urls' )),

##4、在models中的使用
from DjangoUeditor.models import UEditorField
class Blog(models.Model):
	Name=models.CharField(,max_length=100,blank=True)
	Content=UEditorField(u'内容	',width=600, height=300, toolbars="full", imagePath="", filePath="", upload_settings={"imageMaxSize":1204000},
             settings={},command=None,event_handler=myEventHander(),blank=True)



'''