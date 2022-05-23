# Facial-Recognition
A computer vision and artificial intelligence project to control the system of actual door lock system.

## Description

This project is using OpenCV to perform facial recognition in Rock Pi 4B (OS: Debian 10), pyGame for graphic user interface. The door lock system will be controlled by Arduino (UNO R3), Rock Pi 4B and Arduino are communicate to each other via Serial communication using pySerial library.

### Set up
```
$ mkdir dataset
$ mkdir trainer
```

### Requirements
* Library list
1. numpy
2. opencv-python
3. opencv-contrib-python
4. pyserial
5. pandas
6. pygame
```
$ pip install [library name] or pip install -r requirements.txt
```

### Record user data
* The program will automatically take a picture of user face in gray scale and only crop for the face.
* The dataset will be stored in /datasets directory
* Enter ID and user's name in the program. If the ID is already taken, the data will be override.
* The program is take up to 600 pictures of each user, can be pause/resume the program by pressing 'ESC'
```
$ python 01_face_dataset.py
```

### Train model of facial recognition
* The program will use dataset in /datasets and train the facial recognition model for users using LBPH (Local Binary Pattern Histogram) algorithm.
* The model will be saved as /trainer/trainer.xml
```
$ python 02_face_training.py
```

### Run facial recognition
* If you are running on HSH_UI.py, you will be able to use 3 features:
1. Facial recognition to control the door lock
2. Pin password
3. Record new user data

* If you are running on 03_face_recognition.py, you will be able to use only facial recognition to control the door lock.
```
# without GUI
$ python 03_face_recognition.py

# with GUI
$ python HSH_UI.py
```

## Acknowledgment
[Mjrovai(OpenCV-Facial Recognition)](https://github.com/Mjrovai/OpenCV-Face-Recognition)
