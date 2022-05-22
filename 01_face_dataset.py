''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Marcelo Rovai: https://github.com/Mjrovai/OpenCV-Face-Recognition   

Developed by Charnkanit Kaewwong @ 22 May 2022   

'''

import cv2
import os
import pandas as pd

# Find and return the avaliable camera index 
def cam_idx():
    idx = 0
    while idx < 5:
        cam = cv2.VideoCapture(idx)
        if not cam.read()[0]:
            cam.release()
        else:
            cam.release()
            return idx
        idx += 1
    return None

idx = cam_idx()
cam = cv2.VideoCapture(idx)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# set Cascade Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


df = pd.read_csv('name.csv')
names = df['Name'].tolist()
# For each person, enter one numeric face ID in terminal
face_id = input('\n enter user id ==>  ')
face_name = input('\n enter user\'s name ==> ')

# Save user's name to csv file
if (len(names) > int(face_id)): # override name to exist user
    names[int(dace_id)] = face_name
else: 				# add new to lastest name
    face_id = len(names) + 1
    names.append(face_name)
df = pd.DataFrame(names, columns=['Name'])
df.to_csv('name.csv', index=False)

count = 0

while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the /datasets/
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' to pause program from capturing face image
    if k == 27:
        while (True):
            k = cv2.waitKey(100) & 0xff # Press 'ESC' again to resume the program
            if k == 27:
                break
    elif count >= 600: # Take 600 face sample and stop video
         break

print("\n [Finish!] Face dataset stored in /datasets/")
cam.release()
cv2.destroyAllWindows()


