import cv2
import numpy as np

from opencv_test.basic.parking import Parking


def park_detect():
    img = cv2.imread('../image/park.png')
    parking  =   Parking( )
    gray_img = parking.convert_gray_scale(img)


    masked = parking.select_rgb_white_yellow(img)

    parking.cv_show("masked22", masked)
    selectRegion= parking.select_region(masked)
    edged = parking.detect_edges(selectRegion)
    parking.cv_show("selectRegion", selectRegion)
    parking.cv_show("edged", edged)
    parking_lines = parking.hough_lines(edged)
    print(parking_lines)
    # new_img,spot_dict=parking.draw_parking(edged,parking_lines)
    # parking.cv_show("new_img", new_img)

if __name__ == '__main__':
    park_detect()