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
pygame.init()
pygame.camera.init()

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

password_correct = False
run = True

pygame.display.set_caption("Numpad")

# Face recognition set up part start here ---------------

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Aomsin', 'Joey', 'phak', 'Bowasia', 'W'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(1)
cam.set(3, 640) # set video widht 640
cam.set(4, 480) # set video height 480

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Face recognition part end here ---------------
class Password:
    def __init__(self):
        self.password = password
        self.in_pos_x = NUMPAD_X
        self.in_pos_y = NUMPAD_Y


    def print_text(self, text = 'null', posi_x = 0,posi_y = 0, size = 50, color = BLACK):
        font = pygame.font.SysFont('Font.ttf', size)
        text = font.render(str(text), True, color)
        text_center = ((posi_x - (text.get_width()/2.5)),(posi_y - (text.get_height()/2)))
        screen.blit(text, text_center)
        pygame.display.flip()
    
    def draw_numpad(self, in_pos_x = 0, in_pos_y = 0, rad = 45, dis = 2):
        global state
        self.rad = rad
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
                            self.print_text('Cancel', posi_x, posi_y, color = WHITE, size = 35)
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
        
        pygame.display.flip()
    
    def get(self):
        global password_count
        
        mouse_pos = pygame.mouse.get_pos()

        for i in range(11):
            if pos_xy[i][0] - self.rad <= mouse_pos[0] <= pos_xy[i][0] + self.rad and pos_xy[i][1] - self.rad <= mouse_pos[1] <= pos_xy[i][1] + self.rad:
                if i == 10:
                    self.reset()
                else:
                    input_password.append((i+1)%10)
                    password_count += 1
                    print((i+1)%10, password_count, input_password)
        
        for i in range (password_count):
            self.print_text('*', self.in_pos_x + 60 + 30*i, self.in_pos_y - 30, 70, WHITE)
            

        if password_count == 6:
            if input_password == self.password:
                self.print_text('correct', 100, 100, 50)
                password_correct = True
                
            else:
                self.print_text('incorrect', 100, 100, 50)
            print('reset')
            pygame.time.wait(1000)
            self.reset()

    def menu(self):
        screen.blit(BG, (0,0))
        self.print_text("Add person",100,100,50,WHITE)
        pygame.draw.rect(surface,WHITE_TRAN,(100,150,200,100))
        screen.blit(surface, (0,0)) 
        pygame.display.flip()


    def reset(self):
        global password_count
        screen.blit(BG, (0,0))
        password_count = 0
        pos_xy.clear()
        input_password.clear()
        self.draw_numpad(NUMPAD_X,NUMPAD_Y,45)

def camera_vdo(cam_posx = 0, cam_posy = 0, size = 1):
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 55):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
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
    pygame.display.flip()


 
# -------- Main Program ---------------------------------------------------------------------------
password = Password()
password.menu()
#password.draw_numpad(NUMPAD_X,NUMPAD_Y,45)

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cam.release()
            run = False 
        if event.type == pygame.MOUSEBUTTONUP:
            password.get()
        #camera_vdo(50,50,70)

pygame.quit()

