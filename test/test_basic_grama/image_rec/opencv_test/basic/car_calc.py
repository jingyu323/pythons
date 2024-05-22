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

if __name__ == '__main__':
    cal_car()