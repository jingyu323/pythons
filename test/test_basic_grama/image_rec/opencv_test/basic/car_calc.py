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

    mog =cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=False)
    kenel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


    min_w = 60
    min_h = 60
    line_h = 430
    offset = 11
    line_w= int(960/2)
    max_w = 130
    max_h = 130
    carcount=0
    detection_line_length = 950 -10
    detection_line_x = (960 - detection_line_length) / 2
    while True:
        ret,fram = cap.read()
        if ret ==True:
            cars = []
            # 去噪
            gray = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(7,7),5)
            fgmsk =mog.apply(blur,None, -1)
            # 腐蚀
            ercode = cv2.erode(fgmsk,kenel,iterations=2)
            # 膨胀
            dilate =  cv2.dilate(ercode,kenel,iterations=2)
            # 闭运算
            close = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kenel)
            cous, hier =cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(fram,(10,line_h),(950,line_h),(0,0,255),3)

            for cou in cous:
                x,y,w,h =cv2.boundingRect(cou)
                is_valid = (w>= min_w) and (h>= min_h)  and w <= max_w and h <= max_h
                if not  is_valid:
                    continue
                cv2.rectangle(fram,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)

                cp = center(x,y,w,h)
                cv2.circle(fram,(cp),5,(0,0,255),-1)
                cars.append(cp)
            for (x,y) in cars:
                if    y < (line_h+offset)   and    y > (line_h-offset)    and  x > (detection_line_x) and x < (detection_line_x + detection_line_length) :
                    carcount +=1

                    if  y > (line_h-offset) :
                         print("line_h-offset:"+str(line_h-offset))


                cars.remove((x, y))


            cv2.putText(fram,"Vehicle Count:"+str(carcount),(500,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv2.imshow("fram",fram)

        key = cv2.waitKey(50)
        if key == 27:
            break
    print(carcount)
    cap.release()

    cv2.destroyAllWindows()



def cal_car_remove_background2():

    cap  = cv2.VideoCapture("../video/video.mp4")

    mog =cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=True)
    kenel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    min_w = 100
    min_h = 90
    line_h = 600
    offset = 12
    max_w = 160
    max_h = 160
    carcount=0
    line_w=1280
    detec_line_w= int(line_w/2)
    while True:
        ret,fram = cap.read()
        if ret ==True:
            print(fram.shape)
            cars = []
            print(len(cars))
            # 去噪
            gray = cv2.cvtColor(fram,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(7,7),5)
            fgmsk =mog.apply(blur)
            # 腐蚀
            ercode = cv2.erode(fgmsk,kenel,iterations=2)
            # 膨胀
            dilate =  cv2.dilate(ercode,kenel,iterations=2)
            # 闭运算
            close = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kenel)
            cous, hier =cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(fram,(10,line_h),(1270,line_h),(0,0,255),1)

            for cou in cous:
                x,y,w,h =cv2.boundingRect(cou)
                is_valid = (w>= min_w) and (h>= min_h)  and w <= max_w and h <= max_h
                if not  is_valid:
                    continue
                cv2.rectangle(fram,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2)

                cp = center(x,y,w,h)
                cv2.circle(fram,(cp),5,(0,0,255),-1)
                cars.append(cp)
            print("car1 len:"+str(len(cars)))
            for (x,y) in cars:
                if  y >= (line_h-offset) and y<= (line_h+offset)  :
                    carcount +=1
                    print(x,y)
                    if  y > (line_h-offset) :
                         print("line_h-offset:"+str(line_h-offset))

                    print(carcount)
                cars.remove((x, y))

            print("car2 len:" + str(len(cars)))


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