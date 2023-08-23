#recognising the gestures that are being displayed

from calendar import c
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
#The index may need to be adapted to function correctly
cap = cv2.VideoCapture(0)

#modules from mediapipe 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#hands detector 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#create lables for each letter of the alphabet, labels can be adapted to own needs
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

#needed to change letter when logging to console
prev_letter = None

while True:
    #create empty list
    data_aux = []
    #empty list for x coordinates
    x_ = []
    #empty list for y coordinates
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks: 
        #necessary to prevent crash as mediapipe does not process multiple hands
        if len(results.multi_hand_landmarks) > 1:
            cv2.putText(frame, "Can't process two hands", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else: 
            #will draw landmarks and hand connections to the frame
         for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  
                hand_landmarks,  
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
               #loop though each landmark poin on the hand
                for i in range(len(hand_landmarks.landmark)):
                    #retrieves x and y coordinates for landmark
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            #use model to predict gesture being shown 
            prediction = model.predict([np.asarray(data_aux)])
            #take label index and convert to character
            predicted_character = labels_dict[int(prediction[0])]

            
            #log each letter to the console 
            if predicted_character != prev_letter:  
                    prev_letter = predicted_character
                    print(f'{predicted_character}')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()

cv2.destroyAllWindows()
