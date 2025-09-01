import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


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


DATA_SET_PATH = os.path.join('data_set')
words = np.array(['Rahmat', 'Togri', 'Birgalikda', 'Hamma', 'Faqat'])
no_sequences = 30
sequence_length = 30

for word in words:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_SET_PATH, word, str(sequence)))
        except:
            pass


def get_data():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for word in words:     
            for sequence in range(no_sequences):            
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    if frame_num == 0:
                        cv2.waitKey(3000)
                        cv2.putText(image, '{} {}'.format(word, sequence), (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 4, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        cv2.waitKey(3000)
                    else:
                        cv2.imshow('Data Collection', image)
                    if frame_num == sequence_length - 1:
                        cv2.putText(image, '{} {}'.format(word, sequence), (15, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('Data Collection', image)
                        cv2.waitKey(3000)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_SET_PATH, word, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()


get_data()
