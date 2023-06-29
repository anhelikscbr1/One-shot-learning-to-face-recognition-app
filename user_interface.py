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
        self.image = '' 
        self.flag = True
        
        self.get_image()
        self.check_button()
       
    def show(self):
        self.ui.show()

    def get_image(self):
        wid1  = self.ui.widget
        vbox = QVBoxLayout() 
        wid1  = self.ui.widget
        self.ui.pushButton.clicked.connect(self.select_image)
        #vbox.addWidget(wid1)
        self.label = QLabel("")
        vbox.addWidget(self.label)
        wid1.setLayout(vbox)

    def select_image(self):
        fname = QFileDialog.getOpenFileName(self.ui.widget, 'Imagen')
        imagePath = fname[0]
        self.image = imagePath
        print(imagePath)
        pixmap = QPixmap(imagePath)
        #self.label.setPixmap(QPixmap(pixmap))
        scaled_pixmap = pixmap.scaled(350, 350, QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap(scaled_pixmap))
        self.ui.widget.resize(scaled_pixmap.width(), scaled_pixmap.height())
        #self.ui.widget.resize(pixmap.width(),pixmap.height())
    
    def check_button(self):
        vbox2 = QVBoxLayout() 
        wid2 = self.ui.widget_2
        self.ui.pushButton_2.clicked.connect(self.check_on_db)
        self.label2 = QLabel("")
        vbox2.addWidget(self.label2)
        wid2.setLayout(vbox2)

    def check_on_db(self):
        face_embedding = FaceEmbedding()
        imagePath2 = "./people2/" + face_embedding.nw_image_weaviate(face_embedding, self.image, 1, self.flag)
        wid2 = self.ui.widget_2
        #print(imagePath2)
        pixmap2 = QPixmap(imagePath2)
        #self.label.setPixmap(QPixmap(pixmap))
        scaled_pixmap2 = pixmap2.scaled(350, 350, QtCore.Qt.KeepAspectRatio)
        self.label2.setPixmap(QPixmap(scaled_pixmap2))
        self.ui.widget_2.resize(scaled_pixmap2.width(), scaled_pixmap2.height())

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