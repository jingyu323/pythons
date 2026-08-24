import cv2
if __name__ == '__main__':
    car = cv2.imread('../image/car6.jpeg')
    gray = cv2.cvtColor(car, code=cv2.COLOR_BGR2GRAY)
    # 汽车车牌的特征
    car_detector = cv2.CascadeClassifier('../xml/haarcascade_car_plate.xml')
    plates = car_detector.detectMultiScale(gray)
    for x,y,w,h in plates:
        cv2.rectangle(car,pt1=(x,y),pt2=(x+w,y+h),color=[0,0,255],thickness=2)
    cv2.imshow('plate',car)
    cv2.waitKey(0)
    cv2.destroyAllWindows()