from django.shortcuts import render

# Create your views here.
from blog.models import BlogsPost
import datetime
# Create your views here.
def blog_index(request):
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    now = datetime.datetime.now()
    return render(request,'index.html', {'blog_list':blog_list,'now':now})

def index(request):
    blog_list = BlogsPost.objects.all()  # 获取所有数据
    now = datetime.datetime.now()
    return render(request,'index.html', {'blog_list':blog_list,'now':now})
