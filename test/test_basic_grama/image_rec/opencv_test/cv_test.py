import cv2


# https://blog.csdn.net/u010349629/article/details/130663640
def read_image():
    # 读取图像
    img = cv2.imread('../chepai.png')

    # 显示图像
    cv2.imshow('Image', img)
    cv2.waitKey(0)

    # 保存图像
    cv2.imwrite('new_image.jpg', img)

def change_color():
        # 读取图像
        img = cv2.imread('../chepai.png')

        # 将图像转换为灰度空间
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 显示原图和灰度图
        cv2.imshow('Original Image', img)
        cv2.imshow('Gray Image', gray_img)
        cv2.waitKey(0)

        # 保存灰度图
        cv2.imwrite('gray_image.jpg', gray_img)

def resize_img():
    # 读取图像
    img = cv2.imread('../chepai.png')

    # 获取图像的旋转矩阵
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)

    # 进行图像旋转
    rotated_img = cv2.warpAffine(img, M, (cols, rows))

    # 显示原图和旋转后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('Rotated Image', rotated_img)
    cv2.waitKey(0)

    # 保存旋转后的图像
    cv2.imwrite('rotated_image.jpg', rotated_img)

def extrac_img():
    # 读取图像并转换为灰度图
    img = cv2.imread('../chepai.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化SIFT算法对象并提取图像的关键点和描述符
    sift = cv2.SIFT()




    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    # 在图像中绘制关键点
    res_img = cv2.drawKeypoints(img, keypoints, None)

    # 显示原图和特征点标注后的图像
    cv2.imshow('Original Image', img)
    cv2.imshow('SIFT Features', res_img)
    cv2.waitKey(0)

    # 保存特征点标注后的图像
    cv2.imwrite('sift_features.jpg', res_img)

## cv2.VideoCapture(0) 获取摄像头

### cv2.VideoCapture("./")
def video_test():
    cv2.namedWindow("window",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("window",600,480)
    cap = cv2.VideoCapture("./daoyou.mp4")


    while cap.isOpened():
        open, fram = cap.read()
        if not open:
            break
        cv2.imshow('window', fram)
        key = cv2.waitKey(1000//30)  # 不添加wait key 看不到视频

    cap.release()




if __name__ == '__main__':
    video_test()