"""myblog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from blog import views
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('blog/', views.blog_index),
    path('blog/blog_list', views.blog_list),
    # url(r'', views.index),
    url(r'^$', views.index), #设置默认的主页
    url(r'^add/$', views.add, name='add'),
    path('add/<int:a>/<int:b>/', views.add2, name='add2'),
    url(r'^addform', views.addform),
    url(r'^addpostform', views.addpostform),
    url(r'^addpost$', views.addpost),
    url(r'^addpostJson', views.addpostJson),
]
