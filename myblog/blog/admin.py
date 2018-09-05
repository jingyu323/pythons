from django.contrib import admin

from blog.models import BlogsPost

# Register your models here.

'''
list_display用于指定显示的字段
'''
class BlogsPostAdmin(admin.ModelAdmin):
    list_display = ['title', 'body', 'timestamp']

admin.site.register(BlogsPost, BlogsPostAdmin)

