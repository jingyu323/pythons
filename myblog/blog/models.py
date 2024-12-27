from django.db import models
from DjangoUeditor.models import UEditorField

# Create your models here.
'''
blog目录下的models.py文件，这是定义blog数据结构的地

'''

class BlogsPost(models.Model):
    title = models.CharField(max_length=150)  # 博客标题
    # body = models.TextField()  # 博客正文
    timestamp = models.DateTimeField()  # 创建时间
    body = UEditorField(u'内容', width=600, height=300, toolbars="full", imagePath="images/", filePath="files",
                           upload_settings={"imageMaxSize": 1204000},
                           settings={}, command=None, event_handler='', blank=True)