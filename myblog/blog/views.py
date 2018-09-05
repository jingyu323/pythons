from django.shortcuts import render
from django.http import HttpResponse
from blog.tools.addform import AddForm
from  django.http import JsonResponse

# Create your views here.
from blog.models import BlogsPost
import datetime
# Create your views here.
def blog_index(request):
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    now = datetime.datetime.now()
    return render(request,'index.html', {'blog_list':blog_list,'now':now})

def index(request):
    blog_list = BlogsPost.objects.filter(id=1) # 获取所有数据

    now = datetime.datetime.now()
    return render(request,'index.html', {'blog_list':blog_list,'now':now})

def blog_list(request):
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    now = datetime.datetime.now()
    return render(request,'blog_list.html', {'blog_list':blog_list,'now':now})
'''


'''
def add(request):
    a = request.GET['a']
    b = request.GET['b']
    c = int(a)+int(b)
    return HttpResponse(str(c))


def add2(request, a, b):
    c = int(a) + int(b)
    return HttpResponse(str(c))


def addform(request):
    return render(request,'add.html')

def addpostform(request):
    form = AddForm()
    return render(request,'addpost.html', {'form': form})

def addpost(request):
    if request.method == 'POST':  # 当提交表单时

        form = AddForm(request.POST)  # form 包含提交的数据

        if form.is_valid():  # 如果提交的数据合法
            a = form.cleaned_data['a']
            b = form.cleaned_data['b']
            return HttpResponse(str(int(a) + int(b)))

    else:  # 当正常访问时
        form = AddForm()
    return render(request, 'addpost.html', {'form': form})



def addpostJson(request):
    name_dict = {'twz': 'Love python and Django', 'zqxt': 'I am teaching Django'}
    return JsonResponse(name_dict)

