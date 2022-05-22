import cv2
import numpy as np
import random
import pygame
import pygame.camera
from pygame.locals import *
import math
import sys
import os
import numpy as np
import serial
import time
import pandas as pd
from PIL import Image

pygame.init()
pygame.camera.init()

doorstate = True

BLACK = (0,0,0)
WHITE = (255,255,255)
WHITE_TRAN = (255,255,255,100)

BG = pygame.image.load("Background1.jpg")
SCREEN_H = 480
SCREEN_W = 800

NUMPAD_X = 500
NUMPAD_Y = 80

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), vsync=1)
surface = pygame.Surface((SCREEN_W,SCREEN_H), pygame.SRCALPHA)

password = [int(i) for i in str(123456)]
password_count = 0
pos_xy = []
input_password = []

user_count = 0
input_user = []

password_correct = False
run = True
run_get = False
run_menu = True
run_vdo = False
run_getuser = False
addface = False

pygame.display.set_caption("Home Smart Home Security Service")

# Face recognition set up part start here ---------------

def get_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    user = pd.read_csv('name.csv')
    return recognizer, user


cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc


# Initialize and start realtime video capture
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
if idx == None:
    quit()

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids


def act_cam(idx):
    cam = cv2.VideoCapture(idx)
    cam.set(3, 640) # set video widht 640
    cam.set(4, 480) # set video height 480
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    return cam, minW, minH


# Face recognition set up part end here ---------------
try:
    arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=.1)
except:
    arduino = serial.Serial(port='/dev/ttyUSB1', baudrate=9600, timeout=.1)

def write_read(x):
    arduino.write(bytes(x, encoding='utf-8'))
    time.sleep(0.05)

class Password:
    def __init__(self):
        self.password = password
        self.in_pos_x = NUMPAD_X
        self.in_pos_y = NUMPAD_Y
        self.doorstate = doorstate


    def print_text(self, text = 'null', posi_x = 0,posi_y = 0, size = 50, color = BLACK):
        font = pygame.font.SysFont('Font.ttf', size)
        text = font.render(str(text), True, color)
        text_center = ((posi_x - (text.get_width()/2)),(posi_y - (text.get_height()/2)))
        screen.blit(text, text_center)
        pygame.display.flip()
#        pygame.display.update(pygame.Rect(posi_x,posi_y,text.get_width,text.get_height))
    
    def draw_numpad(self, in_pos_x = NUMPAD_X, in_pos_y = NUMPAD_Y, rad = 45, dis = 2):
        global state
        global run_get
        global password_count
        self.rad = rad
        #pos_xy = []
        surface.fill((0,0,0,0))
        screen.blit(BG, (0,0))
        for t in range(2):
            for j in range(4):
                for i in range(3):
                    posi_x = 2*(rad+dis)*i +rad +in_pos_x
                    posi_y = (rad+dis) * 2 * j +rad +in_pos_y

                    if j == 3 and i >= 1 :
                        if t == 0 and i == 1:
                            pygame.draw.circle(surface, WHITE_TRAN, (posi_x,posi_y), rad)
                            # pygame.draw.circle(screen, BLACK, (posi_x,posi_y), rad, width = 3)
                        if t == 1 and i == 1:
                            self.print_text(0, posi_x,posi_y, 50, WHITE)
                            pos_xy.append([posi_x,posi_y])
                        if t == 1 and i == 2: 
                            if password_count == 0:
                                self.print_text('Cancel', posi_x, posi_y, color = WHITE, size = 35)
                            if password_count == 1:
                                self.print_text('del', posi_x, posi_y, color = WHITE, size = 35)
                            pos_xy.append([posi_x,posi_y])

                    elif j != 3:
                        if t == 0:
                            pygame.draw.circle(surface, WHITE_TRAN, (posi_x,posi_y), rad)
                            # pygame.draw.circle(screen, BLACK, (posi_x,posi_y), rad, width = 3)
                        if t == 1:
                            self.print_text(i+(j*3)+1, posi_x, posi_y, 50, WHITE)
                            pos_xy.append([posi_x,posi_y])
            if t == 0:
                screen.blit(surface, (0,0)) 

        if password_count == 0:
            pygame.display.flip()
            #print('flip')

    
    def get(self):
        global password_count
        global run_get
        global pos_xy
        global addface
        while run_get:
            #print('sddface st' + str(addface))
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_pos = pygame.mouse.get_pos()
                    self.print_text('Password', 250, 70, 80,WHITE)

                    for i in range(11):
                        if pos_xy[i][0] - self.rad <= mouse_pos[0] <= pos_xy[i][0] + self.rad and pos_xy[i][1] - self.rad <= mouse_pos[1] <= pos_xy[i][1] + self.rad:
                            if i == 10 and password_count != 0:
                                self.reset_pin()
                                self.print_text('Password', 250, 70, 80,WHITE)
                             
                            elif i == 10 and password_count == 0:
                                run_get = False
                                self.reset_all()

                            else:
                                input_password.append((i+1)%10)
                                password_count += 1
                                print('password', (i+1)%10, password_count, input_password)
                    
                    if password_count == 1:
                        self.draw_numpad()
                        self.print_text('Password', 250, 70, 80,WHITE)
            #            screen.blit(BG, (0,0))
            #            self.print_text('del', pos_xy[10][0], pos_xy[10][1], color = WHITE, size = 35)
                    for i in range (password_count):
                        self.print_text('*', self.in_pos_x + 60 + 30*i, self.in_pos_y - 30, 70, WHITE)
                    
            

                    if password_count == 6:
                        if addface:
                            print('regist')
                            addface = False
                            run_get = False
                            self.regist_user()
                        else:
                            if input_password == self.password:
                                self.print_text('Password correct', 250, 150, 70,WHITE)
                                time.sleep(1)
                                try:
                                    value = write_read('9')
                                    password_correct = True
                                    surface.fill((0,0,0,0))
                                    screen.blit(BG, (0,0))
                                    self.print_text('Unlocked', 400, 240, 80, WHITE)
                                    pygame.time.wait(10000)
                                except:
                                    pass
                #                pygame.time.wait(5000)
                                
                            else:
                                self.print_text('Password incorrect', 250, 150, 70,WHITE)
                                try:
                                    value = write_read('0')
                                except:
                                    pass
                            #print('reset')
                            pygame.time.wait(1000)
                            self.reset_pin()
                            self.print_text('Password', 250, 70, 80,WHITE)


    def menu(self):
        global run_menu 
        global run_get
        global run_vdo
        global addface
        run_menu = True
        surface.fill((0,0,0,0))
        pos_xy_menu = [[250,100],[250,200],[250,300]]
        screen.blit(BG, (0,0))
        self.print_text("Face rec",400,140,50,WHITE)
        pygame.draw.rect(surface,WHITE_TRAN,(250,100,300,80))
        self.print_text("PIN",400,240,50,WHITE)
        pygame.draw.rect(surface,WHITE_TRAN,(250,200,300,80))
        self.print_text("Add face",400,340,50,WHITE)
        pygame.draw.rect(surface,WHITE_TRAN,(250,300,300,80))
        screen.blit(surface, (0,0)) 
        pygame.display.flip()
        print(len(pos_xy))

        mouse_pos_menu = pygame.mouse.get_pos()
        for i in range(3):
            if pos_xy_menu[i][0] <= mouse_pos_menu[0] <= pos_xy_menu[i][0] + 300  and pos_xy_menu[i][1] <= mouse_pos_menu[1] <= pos_xy_menu[i][1] + 80:
                run_menu = False
                if i == 0:
                    run_vdo = True
                if i == 1:
                    run_get = True
                    self.draw_numpad()
                    self.print_text('Password', 250, 70, 80,WHITE)
                if i == 2:
                    addface = True
                    run_get = True
                    self.draw_numpad()
                    self.print_text('Password', 250, 70, 80,WHITE)
                    self.get()
                    #self.regisat_user()

    def reset_pin(self):
        global password_count
        password_count = 0
        user_count = 0
        input_user = []
        password_correct = False
        pos_xy.clear()
        input_password.clear()
        self.draw_numpad(NUMPAD_X,NUMPAD_Y,45)
    
    def reset_all(self):
        global password_count
        global user_count
        #global run_vdo
        #run_vdo = False
        password_count = 0
        user_count = 0
        input_user.clear()
        password_correct = False
        pos_xy.clear()
        input_password.clear()
        self.menu()

    def regist_user(self):
        global idx
        global run_getuser
        run_getuser = True
        face_id = self.getuser() # fucking number from numpad
        if face_id is not None:
            count = 0
            cam,minW, minH = act_cam(idx)
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                    count += 1

                    #cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y: y+h, x:x+w])
                    cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 ,cv2.LINE_AA)
                    #cv2. imshow('image', img)
                    img=cv2.resize(img,(int(640) , int(480)), interpolation = cv2.INTER_AREA)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    img=np.rot90(img) 
                    img=pygame.surfarray.make_surface(img)
                    img=pygame.transform.flip(img,True,False)
                    screen.blit(img, (80,0))
                    pygame.display.flip()
                    
                if count >= 200:
                    break
            cam.release()
            print("training model")
            surface.fill((0,0,0,0))
            screen.blit(BG, (0,0))
            self.print_text('Waiting for training model :)', 400, 240, 70, WHITE)
            pygame.display.flip()
            path = 'dataset'
            faces, ids = getImagesAndLabels(path)
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            recognizer.save('trainer/trainer.yml')
            print("finish train model")
        self.reset_all()


    def getuser(self):
        global user_count
        global run_get
        global pos_xy
        global run_getuser
        global password_count
        print('getuser')
        pos_xy.clear()
        password_count = 0
        self.draw_numpad()  
        self.print_text('Add user', 250, 70, 80,WHITE)
        yn_pos_x = [300,500]
        yn_pos_y = 300
        input_user.clear()
        user_count = 0
        while True and run_getuser:
            print(len(pos_xy))
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:

                    mouse_pos = pygame.mouse.get_pos()
                    print(len(pos_xy), len(mouse_pos))

                    for i in range(11):
                        if pos_xy[i][0] - self.rad <= mouse_pos[0] <= pos_xy[i][0] + self.rad and pos_xy[i][1] - self.rad <= mouse_pos[1] <= pos_xy[i][1] + self.rad:
                            if i == 10 and user_count != 0:
                                self.reset_pin()
            
                            elif i == 10 and user_count == 0:
                                run_getuser = False
                                self.reset_all()

                            else:
                                input_user.append((i+1)%10)
                                user_count += 1
                                #print((i+1)%10, user_count, input_user)
                                #print("qwqwewqewq")
        
                    if user_count == 1:
                        self.print_text(str(input_user[0]), self.in_pos_x + 135, self.in_pos_y - 30, 70, WHITE)
                        surface.fill((0,0,0,0))
                        screen.blit(BG, (0,0))
                        
                        self.print_text('Your input is ' + str(input_user[0]), 400, 150, 80, WHITE)
                        self.print_text('Yes', 300, 300, 70, WHITE)
                        self.print_text('NO', 500, 300, 70, WHITE)
                        while True and run_getuser:
                            print(len(pygame.event.get()))
                            for event in pygame.event.get():
                                print('a for')
                                if event.type == pygame.MOUSEBUTTONUP:
                                    mouse_pos = pygame.mouse.get_pos()
                                    for i in range(2):
                                        if yn_pos_x[i] - self.rad <= mouse_pos[0] <= yn_pos_x[i] + self.rad and yn_pos_y - self.rad <= mouse_pos[1] <= yn_pos_y + self.rad:
                                            if i == 0:
                                                print('input(user) = ' + str(input_user[0]))
                                                """
                                                surface.fill((0,0,0,0))
                                                screen.blit(BG, (0,0))
                                                self.print_text('waiting', 400, 240, 70, WHITE)
                                                pygame.display.flip()
                                                """
                                                run_getuser = False
                                                return input_user[0]
                                            
                                            if i == 1:
                                                user_count = 0
                                                input_user.clear()
                                                pos_xy.clear()
                                                self.getuser()

        return None

    def camera_vdo(self,cam_posx = 0, cam_posy = 0, size = 1):
        global run_vdo
        unlock = False
        cam, minW, minH = act_cam(idx)
        recognizer, user = get_model()
        names = user['Name'].tolist()
        cancel_pos_x = 750
        cancel_pos_y = 450
        self.rad = 45
        while True and run_vdo:
            #global run_vdo
            check = False
            ret, img =cam.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale( 
                    gray,
                    scaleFactor = 1.2,
                    minNeighbors = 5,
                    minSize = (int(minW), int(minH))
                    )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 45):
                    id = names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                    check = True
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                    
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            img=cv2.resize(img,(int(640 *size/100) , int(480 * size/100)), interpolation = cv2.INTER_AREA)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=np.rot90(img) 
            img=pygame.surfarray.make_surface(img)
            img=pygame.transform.flip(img,True,False)
            screen.blit(img, (cam_posx,cam_posy))
            self.print_text('Cancel', cancel_pos_x, cancel_pos_y, color = WHITE, size = 35)
            if (check):
                try:
                    write_read('9')
                    cam.release()
                    print("unlocked")
                    time.sleep(1)
                    unlock = True
                except:
                    pass
            else:
                try:
                    write_read('0')
                    print("locked")
                    unlock = False
                except:
                    pass
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_pos = pygame.mouse.get_pos()
                    #print(mouse_pos)
                    if cancel_pos_x - self.rad <= mouse_pos[0] <= cancel_pos_x + self.rad and cancel_pos_y - self.rad <= mouse_pos[1] <= cancel_pos_y + self.rad:
                        run_vdo = False
                        cam.release()
                        self.reset_all()
            if unlock:
                    surface.fill((0,0,0,0))
                    screen.blit(BG, (0,0))
                    self.print_text('Unlocked', 400, 240, 80, WHITE)
                    pygame.time.wait(10000)
                    cam,minW, minH= act_cam(idx)
                    unlock = False

            pygame.display.flip()


 
# -------- Main Program ---------------------------------------------------------------------------
password = Password()
password.menu()
#password.draw_numpad(NUMPAD_X,NUMPAD_Y,45)
#cam, minW, minH = act_cam(idx)
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            try:
                cam.release()
            except:
                pass
            run = False 
        if event.type == pygame.MOUSEBUTTONUP:
            if run_get:
                password.get()
            if run_menu:
                password.menu()
        if run_vdo:
            #cam, minW, minH = act_cam(idx)
            password.camera_vdo(50,0,100)

pygame.quit()

