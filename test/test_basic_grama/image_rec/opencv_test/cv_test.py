import os

import cv2


# https://blog.csdn.net/u010349629/article/details/130663640
from tqdm import tqdm


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
        cv2.imwrite('image/gray_image.jpg', gray_img)

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
    cv2.destroyAllWindows()

def video_wirte_test():
    cap = cv2.VideoCapture("./daoyou.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # wr = cv2.VideoWriter("test.mp4",cv2.VideoWriter.fourcc(*'mp4v'),30,(640,480))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    out = cv2.VideoWriter("output_video.mp4",fourcc ,fps,(width, height))
    idx = 0

    while cap.isOpened():


        ret, frame = cap.read()
        if not ret:
            break

        # 写入帧到输出视频
        out.write(frame)

        # 显示帧（可选）
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()




def video_wirte_test2():
    cap = cv2.VideoCapture("./daoyou.mp4")
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # wr = cv2.VideoWriter("test.mp4",cv2.VideoWriter.fourcc(*'mp4v'),30,(640,480))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    # out = cv2.VideoWriter("output_video.mp4",fourcc ,fps,(width, height))


    print(height)
    print(width)
    wr = cv2.VideoWriter("test.mp4",fourcc,30,(width, height))

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # 写入帧到输出视频
        wr.write(frame)

        # 显示帧（可选）
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1000 // 30)

    cap.release()
    wr.release()
    cv2.destroyAllWindows()

def crop_video_by_width(input_video_path,out_video_path):
    # 判断视频是否存在
    if not os.path.exists(input_video_path):
        print('输入的视频文件不存在')

    video_read_cap = cv2.VideoCapture(input_video_path)
    input_video_height  = int(video_read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_video_width = int(video_read_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_video_fps  = int(video_read_cap.get(cv2.CAP_PROP_FPS))

    input_video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    out_video_width = 512;
    out_video_height = 512;
    out_video_size = (int(out_video_width), int(out_video_height))

    video_write_cap = cv2.VideoWriter(out_video_path, input_video_fourcc, input_video_fps, out_video_size)
    while video_read_cap.isOpened():
        result, frame = video_read_cap.read()
        if not result:
            break

        # 裁剪到与原视频高度等宽的视频

        diff = int(input_video_height / 7)
        crop_start_index = int(diff)
        crop_end_index = int(diff + int(input_video_height / 7)*5)

        # 参数1 是高度的范围，参数2是宽度的范围
        target = frame[crop_start_index:crop_end_index,  0:int(input_video_width) ]

        # 再resize到512x512
        target = cv2.resize(target, (out_video_width, out_video_height))
        video_write_cap.write(target)
        cv2.imshow('target', target)
        cv2.waitKey(10)

    video_read_cap.release()
    video_write_cap.release()

    cv2.destroyAllWindows()
"""

获取摄像头数据

"""

def videocapture():
    # cap = cv2.VideoCapture(0) # 获取本地摄像头
    # 获取网络摄像头 http://192.168.1.63/ISAPI/Streaming/channels/103/picture  海康
    cap = cv2.VideoCapture('rtsp://admin:xxx@192.168.1.63:554/snl/live/1/1')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # 定义视频对象输出
    writer = cv2.VideoWriter("video_result.mp4", fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('teswell', frame)
        # 调整帧数
        key = cv2.waitKey(24)
        writer.write(frame)
        # 按Q退出
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

"""
把本地图片 缩小

fourcc编码类型不合适
fourcc编码类型与系统安装的编码器有关。一般设置’MJPG’，或者直接用-1采用默认的参数。
frameSize设置不一致
设置frameSize与frame的分辨率一致
"""


def scal_img():
    img = cv2.imread('./image/cat.jpg', cv2.IMREAD_UNCHANGED)

    # 缩小图像
    scale_percent = 25  # 设置缩小比例（例如50表示缩小到原来的一半）
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(img, dim)
    cv2.imshow('test', resized_image)
    cv2.waitKey(0)




if __name__ == '__main__':
    # 判断视频是否存在
    # crop_video_by_width(r'./daoyou.mp4','result.mp4')
    scal_img()