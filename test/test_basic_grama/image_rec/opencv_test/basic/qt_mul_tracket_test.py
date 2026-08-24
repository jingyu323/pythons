
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage


class ROIThread(QThread):
    signal = pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stop_signal = False

    def run(self):
        filename = r'../image/stars.png'
        frame = cv2.imread(filename)

        # 获取 ROI 位置和大小
        roi = cv2.selectROI(frame, fromCenter=False)
        x, y, w, h = roi

        # 绘制 ROI 区域
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 将帧转换为 QImage 格式发送给主线程
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.signal.emit(p)

    def stop(self):
        self.stop_signal = True

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.roi_thread = ROIThread()
        self.roi_thread.signal.connect(self.show_frame)

        self.image_label = QtWidgets.QLabel()

        button_layout = QtWidgets.QHBoxLayout()
        start_button = QtWidgets.QPushButton('Select ROI')
        start_button.clicked.connect(self.start_roi_thread)
        button_layout.addWidget(start_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def start_roi_thread(self):
        self.roi_thread.start()

    def stop_roi_thread(self):
        self.roi_thread.stop()

    def show_frame(self, img):
        self.image_label.setPixmap(QPixmap.fromImage(img))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()