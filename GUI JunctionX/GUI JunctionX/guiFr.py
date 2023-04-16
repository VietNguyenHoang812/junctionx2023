from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPen

import tkinter as tk

# root = tk.Tk()

# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        scale = 4
        MainWindow.resize(scale * 160 * 2 + 30, scale * 90 * 2 + 70)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(10, 50, 160 * scale, 90 * scale))
        self.image.setFrameShape(QtWidgets.QFrame.Box)
        self.image.setText("Camera 1")
        self.image.setObjectName("image")
        self.image1 = QtWidgets.QLabel(self.centralwidget)
        self.image1.setGeometry(QtCore.QRect(160 * scale + 20, 50, 160 * scale, 90 * scale))
        self.image1.setFrameShape(QtWidgets.QFrame.Box)
        self.image1.setText("Camera 2")
        self.image1.setObjectName("image1")
        self.image2 = QtWidgets.QLabel(self.centralwidget)
        self.image2.setGeometry(QtCore.QRect(10, 90 * scale +  60, 160 * scale, 90 * scale))
        self.image2.setFrameShape(QtWidgets.QFrame.Box)
        self.image2.setText("Camera 3")
        self.image2.setObjectName("image2")
        self.image3 = QtWidgets.QLabel(self.centralwidget)
        self.image3.setGeometry(QtCore.QRect(160 * scale + 20, 90 * scale + 60, 160 * scale, 90 * scale))
        self.image3.setFrameShape(QtWidgets.QFrame.Box)
        self.image3.setText("Camera 4")
        self.image3.setObjectName("image3")

        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(15,60, 150,30))
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(160 * scale + 25,60, 150,30))
        self.label1.setFont(font)
        self.label1.setObjectName("label_1")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(15, 90 * scale + 65, 150,30))
        self.label2.setFont(font)
        self.label2.setObjectName("label_2")
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(160 * scale + 25, 90 * scale + 65, 150,30))
        self.label3.setFont(font)
        self.label3.setObjectName("label_3")

        
        self.CheckOutline = QtWidgets.QPushButton(self.centralwidget)
        self.CheckOutline.setGeometry(QtCore.QRect( 15, 10, 111, 25))
        self.CheckOutline.setObjectName("pushButton")
        self.AlertOutline = QtWidgets.QPushButton(self.centralwidget)
        self.AlertOutline.setGeometry(QtCore.QRect(250, 10, 111, 25))
        self.AlertOutline.setObjectName("pushButton_2")
        self.CountOutline = QtWidgets.QPushButton(self.centralwidget)
        self.CountOutline.setGeometry(QtCore.QRect(500, 10, 111, 25))
        self.CountOutline.setObjectName("pushButton_3")


        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow",  "<html><head/><body><p><span style=\" color:#ef2929;\">Camera 1</span></p></body></html>"))
        self.label1.setText(_translate("MainWindow",  "<html><head/><body><p><span style=\" color:#ef2929;\">Camera 2</span></p></body></html>"))
        self.label2.setText(_translate("MainWindow",  "<html><head/><body><p><span style=\" color:#ef2929;\">Camera 3</span></p></body></html>"))
        self.label3.setText(_translate("MainWindow",  "<html><head/><body><p><span style=\" color:#ef2929;\">Camera 4</span></p></body></html>"))
        self.CheckOutline.setText(_translate("MainWindow", "Check Outline"))
        self.AlertOutline.setText(_translate("MainWindow", "Alert Outline"))
        self.CountOutline.setText(_translate("MainWindow", "Count Outline"))