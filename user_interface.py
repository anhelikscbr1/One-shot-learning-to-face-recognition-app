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
        self.ui.pushButton_2.clicked.connect(self.check_on_db)
        self.image = '' 
        wid1  = self.ui.widget
        vbox = QVBoxLayout() 
        self.ui.pushButton.clicked.connect(self.select_image)
        #vbox.addWidget(wid1)
        self.label = QLabel("")
        vbox.addWidget(self.label)
        wid1.setLayout(vbox)

    def show(self):
        self.ui.show()

    def select_image(self):
       print("Hola")
       self.getImage()

    def getImage(self):


        fname = QFileDialog.getOpenFileName(self.ui.widget, 'Imagen')
 
        imagePath = fname[0]
        self.image = imagePath
        print(imagePath)
        pixmap = QPixmap(imagePath)
        #self.label.setPixmap(QPixmap(pixmap))
        scaled_pixmap = pixmap.scaled(360, 360, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap(scaled_pixmap))
        self.ui.widget.resize(scaled_pixmap.width(), scaled_pixmap.height())
        #self.ui.widget.resize(pixmap.width(),pixmap.height())

 


    def check_on_db(self):
        vbox2 = QVBoxLayout() 
        self.label = QLabel("Image2")
        vbox2.addWidget(self.label)
        face_embedding = FaceEmbedding()
        imagePath2 = "./people2/" + face_embedding.nw_image_weaviate(face_embedding, self.image, 1)
        wid2 = self.ui.widget_2
        print(imagePath2)
        pixmap2 = QPixmap(imagePath2)
        #self.label.setPixmap(QPixmap(pixmap))
        scaled_pixmap2 = pixmap2.scaled(360, 360, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap(scaled_pixmap2))
        wid2.resize(scaled_pixmap2.width(), scaled_pixmap2.height())
        self.ui.widget_2.setLayout(vbox2)

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
 
        self.label = QLabel("")
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