from PySide6.QtWidgets import QWidget,QPushButton,QHBoxLayout,QVBoxLayout ,QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit
from PySide6.QtGui import QPixmap

from PySide6 import QtCore, QtWidgets
from PySide6.QtUiTools import QUiLoader

from face_embedding_manipulation import FaceEmbedding

import sys

loader = QUiLoader()

class UserInterface(QtCore.QObject): #An object wrapping around our ui
    def __init__(self):
        super().__init__()
        self.ui = loader.load("ui_model.ui", None)
        self.ui.setWindowTitle("One-Shot Learning Vector Database APP")
        self.ui.pushButton.clicked.connect(self.select_image)
        self.ui.pushButton_2.clicked.connect(self.check_on_db)
        self.image = ''

    def show(self):
        self.ui.show()

    def select_image(self):
        vbox = QVBoxLayout() 
        self.label = QLabel("Hello")
        vbox.addWidget(self.label)
        print("Hola")
        self.getImage()
        self.ui.widget.setLayout(vbox)

    def getImage(self):
        wid1 = self.ui.widget
        fname = QFileDialog.getOpenFileName(wid1, 'Imagen')
 
        imagePath = fname[0]
        self.image = imagePath
        print(imagePath)
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        scaled_pixmap = pixmap.scaled(360, 360, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap(scaled_pixmap))
        wid1.resize(scaled_pixmap.width(), scaled_pixmap.height())

    def check_on_db(self):
        vbox = QVBoxLayout() 
        self.label = QLabel("Hello")
        vbox.addWidget(self.label)
        face_embedding = FaceEmbedding()
        imagePath = "./people2/" + face_embedding.nw_image_weaviate(face_embedding, self.image, 1)
        wid2 = self.ui.widget_2
        print(imagePath)
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        scaled_pixmap = pixmap.scaled(360, 360, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap(scaled_pixmap))
        wid2.resize(scaled_pixmap.width(), scaled_pixmap.height())
        self.ui.widget_2.setLayout(vbox)

class Window(QWidget):
    def __init__(self):
        super().__init__()
 
        self.title = "PyQt6 Open File"
        self.top = 200
        self.left = 500
        self.width = 1400
        self.height = 1300
 
        self.InitWindow()
 
    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        vbox = QVBoxLayout()
 
        self.btn1 = QPushButton("Open Image")
        self.btn1.clicked.connect(self.getImage)
 
        vbox.addWidget(self.btn1)
 
        self.label = QLabel("Hello")
        vbox.addWidget(self.label)
 
        self.setLayout(vbox)
 
 
    def getImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file')
 
        imagePath = fname[0]
        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(),pixmap.height())
#App = QApplication(sys.argv)
#window = Window()
#window.show()
#sys.exit(App.exec_())