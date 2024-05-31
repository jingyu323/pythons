import cv2

def  show_img(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def card_recgnize():
    tmplate = cv2.imread('../image/tmplate.jpg')

    tm_gray = cv2.cvtColor(tmplate, cv2.COLOR_BGR2GRAY)
    # 二值化 灰度化之后
    tm_ref = cv2.threshold(tm_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    counters,her =cv2.findContours(tm_ref,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for count in counters:
        cv2.drawContours(tmplate,[count],-1,(0,0,255),-1)

    show_img("tmplate", tmplate)

    card = cv2.imread('../image/card.png')
    card_gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    card_recgnize()
