# UzSignAI

**UzSignAI** is a real-time gesture recognition system built with **OpenCV**, **Mediapipe**, and **TensorFlow/Keras**.  
The project is focused on recognizing hand gestures in Uzbek Sign Language using a webcam.

## Features

- Real-time gesture recognition using a webcam
- Recognition of predefined gestures: 'Rahmat', 'Togri', 'Birgalikda', 'Hamma', 'Faqat'
- Data collection script to capture and save training data
- Model training script to train a deep learning model using LSTM networks

## Installation
- Clone the Repository
```bash
git clone https://github.com/me-sarvar/uz-sign-ai.git
cd uz-sign-ai
```
- Install Required Dependencies
```bash
pip install -r requirements.txt
```
## Project Structure

- `main.py`: The main script for real-time gesture recognition.
- `data_collect.py`: Script to collect training data by capturing hand gestures.
- `model_train.py`: Script to train a deep learning model using LSTM networks.

## Usage

### Real-time Gesture Recognition

1. Run the `main.py` script to start the application.
2. The webcam will be activated, and the application will recognize and display gestures in real-time.

### Data Collection

1. Run the `data_collect.py` script to capture training data for hand gestures.
2. Follow the instructions displayed on the screen to perform gestures for each word in the predefined vocabulary.
3. The captured data will be saved in the `data_set` directory.

### Model Training

1. Run the `model_train.py` script to train the deep learning model.
2. The model will be saved in the `models` directory.

## Configuration

- The predefined gestures are stored in the `words` array in both `main.py` and `data_collect.py`.
- Model parameters and architecture can be adjusted in `model_train.py` under the `create_model` function.

## Notes

- The application is set up to recognize gestures defined in the `words` array.
- Training data is saved in the `data_set` directory, and the trained model is saved in the `models` directory.
- Feel free to customize the project according to your needs and enjoy experimenting with gesture recognition!
