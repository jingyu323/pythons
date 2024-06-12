import cv2

from opencv_test.basic.parking import Parking


def park_detect():
    img = cv2.imread('../image/park.png')
    parking  =   Parking( )
    gray_img = parking.convert_gray_scale(img)

    masked = parking.select_rgb_white_yellow(img)
    parking.cv_show("masked",masked)

    edged = parking.detect_edges(masked)
    parking.select_region(gray_img)

if __name__ == '__main__':
    park_detect()