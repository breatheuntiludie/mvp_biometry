import cv2
import sys
import pickle
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, QTimer
from PyQt5.QtWidgets import QLabel, QWidget, QPushButton, QVBoxLayout, QApplication, QHBoxLayout, QMessageBox
import pandas as pd
from deepface import DeepFace
import os

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') #cascade for face detection
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#initialise model + dataset
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",]
# we have 3 different metrics, but we use only euclidean
metrics = ["cosine", "euclidean", "euclidean_l2"]


class Camera:

    def __init__(self, camera):
        self.camera = camera
        self.cap = None
        self.image_recog = "null"
        self.image_ref = "null"
        self.path_name = "null"
        self.counter = 1
        self.check = 0
        self.id = -1

    def openCamera(self):
        #print('openCamera class')
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # set width webcam
        self.cap.set(4, 480)  # set height webcam
        #print('OpenCamera after cap.set')
        if not self.cap.isOpened():
            print('failure')
            msgBox = QMessageBox()
            msgBox.setText("Failed to open camera.")
            msgBox.exec_()
            return -2 # error of webcam

    def initialize(self):
        self.cap = cv2.VideoCapture(self.camera)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, camera = None):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 1000)
        MainWindow.setStyleSheet("background-color: rgb(17, 143, 202);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Button_1 = QtWidgets.QPushButton(self.centralwidget)
        self.Button_1.setGeometry(QtCore.QRect(910, 670, 171, 61))
        self.Button_1.setStyleSheet("background-color: rgb(0, 255, 127);")
        #self.Button_1.setObjectName("Button_1")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1100, 670, 171, 61))
        self.pushButton.setStyleSheet("background-color: rgb(170, 0, 0);")
        #self.pushButton.setObjectName("pushButton")

        self.pushButton_face = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_face.setGeometry(QtCore.QRect(130, 730, 171, 61))
        self.pushButton_face.setStyleSheet("background-color: rgb(170, 40, 30);")

        MainWindow.setCentralWidget(self.centralwidget)
        #self.menubar = QtWidgets.QMenuBar(MainWindow)
        #self.menubar.setGeometry(QtCore.QRect(0, 0, 1500, 21))
        #self.menubar.setObjectName("menubar")
        #MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        #self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Add a label for video
        self.label = QtWidgets.QLabel(self.centralwidget)
        #self.label = QLabel()
        self.label.setGeometry(QtCore.QRect(830, 180, 640, 480))
        #self.label.setFixedSize(640, 480)
        #MainWindow.addWidget(self.label)

        # Add the label for shoplifters (recognised)
        self.label_recog = QtWidgets.QLabel(self.centralwidget)
        self.label_recog.setGeometry(QtCore.QRect(100,70, 200,200))
        self.label_ref = QtWidgets.QLabel(self.centralwidget)
        self.label_ref.setGeometry(QtCore.QRect(300,70, 200,200))


        self.camera = camera
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        #self.timer.timeout.connect(self.show_pic)

        self.add_functions()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Биометрическая система идентификации посетитиелей магазина"))
        self.Button_1.setText(_translate("MainWindow", "Просмотр камеры"))
        self.pushButton.setText(_translate("MainWindow", "Прекратить запись"))
        self.pushButton_face.setText(_translate("MainWindow", "Показать лица"))

    def add_functions(self):
        self.Button_1.clicked.connect(self.start_video)
        self.pushButton.clicked.connect(self.end_video)
        self.pushButton_face.clicked.connect(self.show_pic)

    def show_pic(self):
        if camera.image_recog != "s":
            print(camera.image_recog)
            pixmap_2 = QPixmap(camera.image_recog).scaled(200,270)
            pixmap_3 = QPixmap(camera.image_ref)
            # face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            print(pixmap_2)
            self.label_recog.setPixmap(pixmap_2)
            self.label_ref.setPixmap(pixmap_3)
    def start_video(self):
        #thieves = pd.read_pickle('C:/Users/user/Desktop/face_recog_gui_mvp/thieves.pkl')
        delete_file = 'C:/Users/user/Desktop/face_recog_gui_mvp/db/representations_facenet512.pkl'
        if os.path.isfile(delete_file):
            os.remove(delete_file)
        else:
            print('Path is not a file')
        print('start video')
        camera.openCamera()
        print('camera was opened')
        self.timer.start(1000. / 24)

    def nextFrameSlot(self):
        ret, frame = camera.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:
            #print(x, y, h, w)
            roi_gray = gray[y:y + h, x:x + w]
            img_save = "new_pic/" + "1.png"
            cv2.imwrite(img_save, roi_gray)
            #crashes if your photo can't be recognised as face, set enforce_detection = False
            df = DeepFace.find(img_path=img_save, db_path="C:/Users/user/Desktop/face_recog_gui_mvp/db",
                               model_name=models[2], distance_metric=metrics[1], enforce_detection = False,
                               prog_bar = True, silent = True)
            width = x + w
            height = y + h
            if df.empty:
                print("no matches")
                name = "unknown"
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (width, height), color, stroke)
            else:
                camera.image_ref = img_save
                camera.image_recog = df.iloc[0]['identity']
                name = "thief"
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (width, height), color, stroke)
                #name = (df.iloc[0]['identity']).split("/")[6]
            print(camera.image_recog)
            delete_file = 'C:/Users/user/Desktop/face_recog_gui_mvp/db/representations_facenet512.pkl'
            if os.path.isfile(delete_file):
                os.remove(delete_file)
                print("deleted")
            else:
                print('Path is not a file')

        #print('Blue border was made')
        image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        #print('QImage?')
        pixmap = QPixmap.fromImage(image)
        #print('QPixmap?')
        self.label.setPixmap(pixmap)
        #print('setpixmap?')


    def end_video(self):
        #if cv2.waitKey(20) & 0xFF == ord('q'):
            #return 0
        #camera.cap.release(0)
        self.timer.stop()
        self.label.clear()
        #cv2.destroyAllWindows()

camera = Camera(0)

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()
#sys.exit(app.exec_())
app.exit(app.exec_())