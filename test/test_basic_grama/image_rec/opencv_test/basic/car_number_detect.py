import cv2


def car_number_detect(img):
    car = cv2.imread(img)
    cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
    car_gray = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)



if __name__ == '__main__':
    image = "../../chepai.png"
    car_number_detect(image);