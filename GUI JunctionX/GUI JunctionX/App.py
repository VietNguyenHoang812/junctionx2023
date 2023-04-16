import sys
# pip install pyqt5
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
import os
import time

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import *
from guiFr import Ui_MainWindow

class Menu(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Title")
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.CheckOutline.clicked.connect(self.check_outline)
        self.uic.AlertOutline.clicked.connect(self.alert_outline)
        self.uic.CountOutline.clicked.connect(self.count_outline)

        timer = QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(10)

        self.update_image()
    
    def check_outline(self):
        pass

    def alert_outline(self):
        pass

    def count_outline(self):
        pass
    
    def update_image(self):
        self.displayImg('/home/hoangminh/Documents/hackathon_overlapping_area_estimation/SuperGluePretrainedNetwork/output_demo/0.jpg')
        self.displayImg1('/home/hoangminh/Documents/hackathon_overlapping_area_estimation/SuperGluePretrainedNetwork/output_demo/1.jpg')
        self.displayImg2('/home/hoangminh/Documents/hackathon_overlapping_area_estimation/SuperGluePretrainedNetwork/output_demo/2.jpg')
        self.displayImg3('/home/hoangminh/Documents/hackathon_overlapping_area_estimation/SuperGluePretrainedNetwork/output_demo/3.jpg')

    def displayImg(self, img):
        qt_img = self.convert_cv_qt(img)
        self.uic.image.setPixmap(qt_img)

    def displayImg1(self, img):
        qt_img = self.convert_cv_qt(img)
        self.uic.image1.setPixmap(qt_img)

    def displayImg2(self, img):
        qt_img = self.convert_cv_qt(img)
        self.uic.image2.setPixmap(qt_img)

    def displayImg3(self, img):
        qt_img = self.convert_cv_qt(img)
        self.uic.image3.setPixmap(qt_img)

    def convert_cv_qt(self, path):
        """Convert from an opencv image to QPixmap"""
        cv_img = cv2.imread(path)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 640, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Menu()
    ex.show()
    sys.exit(app.exec_())
