import cv2
import keras
import numpy as np

from opencv_test.basic.parking import Parking


def park_detect():
    img = cv2.imread('../image/park.png')
    parking  =   Parking( )
    gray_img = parking.convert_gray_scale(img)


    masked = parking.select_rgb_white_yellow(img)

    parking.cv_show("masked22", masked)
    selectRegion= parking.select_region(gray_img)
    edged = parking.detect_edges(selectRegion)
    parking.cv_show("selectRegion", selectRegion)
    parking.cv_show("edged", edged)
    parking_lines = parking.hough_lines(edged)
    print(parking_lines)
    new_img =parking.draw_lines(edged,parking_lines)
    parking.cv_show("new_img", new_img)
    block_images,rects =parking.identify_blocks(img,lines=parking_lines)
    parking.cv_show("block_images", block_images)

    new_image, spot_dict=parking.draw_parking(img,rects=rects,thickness=1)

    parking.cv_show("new_image", new_image)

    spot_map_img=parking.assign_spots_map(new_image,spot_dict)

    parking.cv_show("spot_map_img", spot_map_img)

    parking.save_images_for_cnn(img,spot_dict)
    class_dictionary={}
    class_dictionary[0]="empty"
    class_dictionary[1]="occupied"

    weifgts_path="car.h5"

    model= keras.models.load_model(weifgts_path)



    parking.predict_on_image(img, spot_dict, model, class_dictionary,)

if __name__ == '__main__':
    park_detect()