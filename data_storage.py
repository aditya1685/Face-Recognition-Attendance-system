import cv2
import numpy as np
import os
import pickle 

cap = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_data = []
name = input('Enter your name: ')
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w, :]
        resized_image = cv2.resize(crop_image, (50, 50))
        
        if len(face_data) < 50 and i % 10 == 0:
            face_data.append(resized_image)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    cv2.imshow('frame', frame)
    i += 1
    
    if len(face_data) == 50:
        break

cap.release() 
cv2.destroyAllWindows()

face_data = np.array(face_data)
face_data = face_data.reshape(50, -1)

if not os.path.exists('attendance'):
    os.makedirs('attendance')

if 'names.pkl' not in os.listdir('attendance/'):
    names = [name] * 50
    with open('attendance/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('attendance/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 50
    with open('attendance/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'face_data.pkl' not in os.listdir('attendance/'):
    with open('attendance/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('attendance/face_data.pkl', 'rb') as f:
        existing_data = pickle.load(f)
    faces = np.vstack((existing_data, face_data))
    with open('attendance/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

