#creating and formatting a dataset from the collected images
from ctypes.wintypes import HANDLE
import os
import pickle
#pip install mediapipe
import mediapipe as mp
#pip install OpenCV-Python
import cv2
#plot how images look
import matplotlib.pyplot as plt

#for detecting landmarks to draw landmarks on top of images 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#hands detector 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data =[]
labels = []

#iterate directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        #need to convert image into RGB to input image into mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)


#save all data 
f = open('data.pickle', 'wb') 
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
