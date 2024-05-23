import cv2

def cal_car():

    cap  = cv2.VideoCapture("../video/cheliu.mp4")

    while True:
        ret,fram = cap.read()
        if ret ==True:
            cv2.imshow("video",fram)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

def center( x,y,w,h):
    x1 = int(w/2)
    y1= int(h/2)
    cx = int(x) +x1
    cy = int(y) +y1
    return  cx,cy


def cal_car_remove_background():

    cap  = cv2.VideoCapture("../video/gaosu.mp4")

    mog =cv2.createBackgroundSubtractorMOG2()
    kenel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    min_w = 90
    min_h = 80
    line_h = 450
    offset = 20
    cars=[]
    carcount=0
    while True:
        ret,fram = cap.read()
        if ret ==True:
            # 去噪
            gray = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(7,7),5)
            fgmsk =mog.apply(blur)
            # 腐蚀
            ercode = cv2.erode(fgmsk,kenel)
            # 膨胀
            dilate =  cv2.dilate(ercode,kenel)
            # 闭运算
            close = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kenel)
            cous, hier =cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(fram,(10,line_h),(950,line_h),(0,0,255),3)

            for cou in cous:
                x,y,w,h =cv2.boundingRect(cou)
                is_valid = (w>= min_w) and (h>= min_h)
                if not  is_valid:
                    continue
                cv2.rectangle(fram,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)

                cp = center(x,y,w,h)
                cv2.circle(fram,(cp),5,(0,0,255),-1)
                cars.append(cp)

                for (x,y) in cars:
                    if y > (line_h-offset) and y < (line_h+offset):
                        carcount +=1
                        print(x,y)

                        print(carcount)
                    cars.remove((x, y))

            cv2.putText(fram,"Vehicle Count:"+str(carcount),(500,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv2.imshow("fram",fram)

        key = cv2.waitKey(50)
        if key == 27:
            break
    print(carcount)
    cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    cal_car_remove_background()