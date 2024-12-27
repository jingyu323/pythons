import numpy as np
import cv2
from easyocr_demo import Reader

# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# 1、加载图片
image = cv2.imread('../image/2.jpg')
# 灰度化处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 去除噪声
gray = cv2.bilateralFilter(gray, 0, 17, 17)
# 发现边界
edged = cv2.Canny(gray, 170, 200)
# 2、找到轮廓，根据轮廓面积，排序，提取面积最大的30个
cnts, new  = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
# 3、循环找到最有可能是车牌的轮廓
NumberPlateCnt = None
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # 选择4个角的轮廓，最有可能是车牌
            NumberPlateCnt = approx #这就是车牌索引
            break
# 4、绘制轮廓

(x, y, w, h) = cv2.boundingRect(NumberPlateCnt)
license_plate = gray[y:y + h, x:x + w]


reader = Reader(['en'])
# detect the text from the license plate
detection = reader.readtext(license_plate)
print(detection)

if len(detection) == 0:
    # if the text couldn't be read, show a custom message
    text = "Impossible to read the text from the license plate"
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 3)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
else:
    # draw the contour and write the detected text on the image
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    textlist=[]
    for dec in detection:
        textlist.append(dec[1])

    text = " ".join(textlist)

    print(text)
    cv2.putText(image, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # display the license plate and the output image
    cv2.imshow('license plate', license_plate)
    cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()