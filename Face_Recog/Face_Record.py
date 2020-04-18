# Importing necessary libraries for our program to run
import os  # To store or retrieve Face details

import cv2  # To capture video and detect face in it
import numpy as np  # For calculation and manipulating face details
from tkinter import *

# insert at 1, 0 is the script path (or '' in REPL)

# Haarcascade Classifier for frontal face which will detect your face and gives the (x,y) coordinates along with
# Height and width of your face from the input image

def facerec(name):
    path = "Files/Face_Recognition/"
    classifier = cv2.CascadeClassifier(path + "haarcascade_frontalface_default.xml")

    # Initialize the front camera of your laptop as 0 stands for inbuilt camera and it will start recording
    cap = cv2.VideoCapture(0)

    face_list = []
    count = 50
    while count:

        areas = []
        ret, image = cap.read()  # It will take the image from front camera
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = classifier.detectMultiScale(gray)
        for face in faces:
            x, y, w, h = face
            area = w * h
            areas.append((area, face))
        areas = sorted(areas, reverse=True)
        if len(areas) > 0:
            face = areas[0][1]
            x, y, w, h = face
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (100, 100))
            cv2.imshow("FACE", face_img)
            face_list.append(face_img.flatten())
            count -= 1
            print("{} Record Entered".format(50 - count))
        if cv2.waitKey(1) > 30:
            break

    face_list = np.array(face_list)

    name_list = np.full((len(face_list), 1), name)
    Final = np.hstack([name_list, face_list])
    if os.path.exists(path + "Face_Data.npy"):
        old_records = np.load(path + "Face_Data.npy")
        Final = np.vstack([old_records, Final])
    np.save(path + "Face_Data.npy", Final)
    cap.release()
    cv2.destroyAllWindows()
