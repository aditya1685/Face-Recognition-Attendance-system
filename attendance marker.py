import cv2
import numpy as np
import os
import pickle
import csv
import time
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

cap = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with open('attendance/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('attendance/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imagebackground = cv2.imread(r"C:\Users\ASUS\OneDrive\Desktop\Computer vision\attendance\blank-white-7sn5o1woonmklx1h.jpg")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w, :]
        resized_image = cv2.resize(crop_image, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_image)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        csv_filename = 'attendance/Attendance_' + date + ".csv"
        exist = os.path.isfile(csv_filename)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

        attendance = [str(output[0]), str(timestamp)]

        imagebackground[162:162+480, 55:55+640] = frame

    cv2.imshow('frame', imagebackground)

    k = cv2.waitKey(1)
    
    if k == ord('q'):
        time.sleep(5)

        if exist:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                
    if k == ord('o'):
        break

cap.release()
cv2.destroyAllWindows()
