import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


MODELS_PATH = os.path.join('models')
DATA_SET_PATH = os.path.join('data_set')
actions = np.array(['Rahmat', 'Togri', 'Birgalikda', 'Hamma', 'Faqat'])
no_sequences = 30  
sequence_length = 30  

label_map = {label: num for num, label in enumerate(actions)}
def get_res(action, sequence, frame_num):
    res = np.load(os.path.join(DATA_SET_PATH, action, str(sequence), "{}.npy".format(frame_num)))
    return res
sequences, labels = [], []

for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_SET_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_SET_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def train_model(model, X_train, y_train):
    tb_callback = TensorBoard()
    model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
    print(model.summary())
    model.save(os.path.join(MODELS_PATH, 'model.h5'))


def create_and_train_model(X_train, y_train):
    model = create_model()
    train_model(model, X_train, y_train)
create_and_train_model(X_train, y_train)