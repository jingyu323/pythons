import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread('./img4.jpg',flags=cv2.IMREAD_GRAYSCALE)
    result = cv2.morphologyEx(img,
                     op =cv2.MORPH_CLOSE,
                     kernel = np.ones(shape = [8,8],dtype=np.uint8),
                     iterations=1)

    cv2.imshow('raw',img)
    cv2.imshow('close',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()