import cv2


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 截取图像
def img_extract(img):

    result = img[0:200, 0:200]
    cv_show('result', result)
    print('result', img.shape)
    print('像素点个数',  img.size)
    print('数据类型',  img.dtype)

    # 保存
    # cv2.imwrite('xxx.jpg', img)

def read_video(video_path):
    vc = cv2.VideoCapture(video_path)
    if vc.isOpened():
        open, frame = vc.read()
    else:
        open = False
    # 循环播放每一帧图像
    while open:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('result', gray)
            if cv2.waitKey(10) & 0xFF == 27:  # 27为键盘上的退出键
                break
    vc.release()
    cv2.destroyAllWIndows()


def border_fill(img):
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)

    # 复制法，复制最边缘像素
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    cv_show('replicate', replicate)
    # 反射法，对感兴趣的像素在两边复制，如：fedcba|abcdefgh|hgfedcb
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
    cv_show('reflect', reflect)
    # 反射法，以最边缘像素为轴对称，如：gfedcb|abcdefgh|gfedcba
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                                    borderType=cv2.BORDER_REFLECT_101)
    cv_show('reflect101', reflect101)

    # 外包法，如：cdefgh|abcdefgh|abcdefg
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
    cv_show('wrap', wrap)
    # 常量法，用常数填充
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT,
                                  value=0)

    cv_show('constant', constant)


def  img_yuzhi(img):
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 超过阈值部分取maxval(最大值)，否则取0
    ret,thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    cv_show('thresh1', thresh1)
    # 上面的反转
    ret,thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)  # INV为inverse

    cv_show('thresh2', thresh2)
    # 大于阈值部分为阈值，其他不变
    ret,thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    cv_show('thresh3', thresh3)
    # 大于阈值的不变，其他为0
    ret,thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    cv_show('thresh4', thresh4)
    # 上面的反转
    ret,thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
    # 127为阈值，255为最大值
    cv_show('thresh5', thresh5)


if __name__ == '__main__':
    img = cv2.imread("../image/jinmao.jpg")
    # img_extract(img)

    img_yuzhi(img)
