import datetime
import os
import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
MODELS_PATH = os.path.join('models')
RESULT_PATH = 'results'
os.makedirs(RESULT_PATH, exist_ok=True)
model = load_model(os.path.join(MODELS_PATH, 'model.h5'))

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3),
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 128, 128), thickness=3, circle_radius=3),
                              mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=3, circle_radius=3)
                              )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, face, lh, rh])

def on_exit():    
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()
    save_sentences()

def save_sentences():
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"results_{timestamp}.txt"    
    file_path = os.path.join(RESULT_PATH, file_name)
    with open(file_path, 'w') as file:
        file.write('\n'.join(sentence))

sequence = []
sentence = []

def update_frame():    
    global sequence, sentence

    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        draw_styled_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            with np.printoptions(threshold=np.inf, suppress=True):  # Disable printing step info
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            print(words[np.argmax(res)])  # Add this line
            
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if words[np.argmax(res)] != sentence[-1]:
                        sentence.append(words[np.argmax(res)])
                else:
                    sentence.append(words[np.argmax(res)])
            
            if len(sentence) > 5: 
                sentence = sentence[-5:]         
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)        
        
        label.configure(image=image_tk)
        label.image = image_tk
       
        text_box.delete(1.0, tk.END)
        text_box.insert(tk.END, ' '.join(sentence)) 
        text_box.configure(font=("Monospace", 13))
    
    window.after(10, update_frame)

window = tk.Tk()
window.title("Gesture Recognition")
window.protocol("WM_DELETE_WINDOW", on_exit)

label = tk.Label(window)
label.pack()

text_box = tk.Text(window, height=1, width=60, font=("Monospace", 13))
text_box.pack(side=tk.TOP)

cap = cv2.VideoCapture(0)

holistic = mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.5)

words = np.array(['Rahmat', 'Togri', 'Birgalikda', 'Hamma', 'Faqat'])
threshold = 0.80

update_frame()

window.mainloop()
