
import os
import sys
# import easyocr
# reader = easyocr.Reader(['ch_sim','en'], gpu=True,download_enabled=True) # this needs to run only once to load the model into memory
# reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
from easyocr import easyocr

reader = easyocr.Reader(['ch_sim', 'en'], gpu=True,download_enabled=False)
result = reader.readtext('test.png')
# result = reader.readtext('test2.jpg')
# result = reader.readtext('chepai.png' ,detail = 0)
for res in result:
     print(res)
print("fff")
# 图像作为 numpy 数组（来自 opencv）传递



# 如何提高识别度