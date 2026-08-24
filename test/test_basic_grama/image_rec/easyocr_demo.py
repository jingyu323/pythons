import cv2
import easyocr
# reader = easyocr.Reader(['ch_sim','en'], gpu=True,download_enabled=True) # this needs to run only once to load the model into memory
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
# reader = easyocr.Reader([ 'ch_tra','en'], gpu=False,download_enabled=True)
result = reader.readtext('test.png')
# result = reader.readtext('test2.jpg')
# result = reader.readtext('chepai.png' ,min_size=1,detail = 0)
for res in result:
     print(res)
print("\n")
# 图像作为 numpy 数组（来自 opencv）传递
img = cv2.imread('chepai.png')
result = reader.readtext(img)

for res in result:
     print(res)


# 如何提高识别度