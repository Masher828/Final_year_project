import numpy as np
import cv2
import pandas
import sys
classifier = cv2.CascadeClassifier("Files/Face_Recognition/haarcascade_frontalface_default.xml")
cap= cv2.VideoCapture(0)
while True :
        areas=[]
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(image)
        for face in faces:
            x, y, w, h = face
            area = w * h
            areas.append((area, face))
        areas = sorted(areas, reverse=True)
        if len(areas) > 0:
            face = areas[0][1]
            x, y, w, h = face
            face_img = gray[y:y + h, x:x + w]
            gray = cv2.resize(face_img, (100, 100))
            face_data = gray.flatten().reshape(1, -1)

            name = "Manish"

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv2.putText(image, name[0], (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, 255)

        cv2.imshow("Face", image)

        if cv2.waitKey(1) > 30:
            break
        cap.release()
        cv2.destroyAllWindows()