# Real-Time Sign Language Recognition using Mediapipe and LSTMs

This project implements a **sign language recognition system** using **Mediapipe**, **TensorFlow/Keras**, and **OpenCV**. It enables you to **collect gesture data**, **train a deep learning model**, and **predict signs in real time** through your webcam.  

The system supports both **Mediapipe Holistic (pose + hands + face)** and **Mediapipe Hands (hands only)** modes.

---

## Features
- Collects sign language data using webcam  
- Sequential modeling of gestures with **LSTMs**  
- Three operating modes:
  - `collect` → Gather training data  
  - `train` → Train an action recognition model  
  - `predict` → Perform real-time sign predictions  
- Stores dataset automatically in structured folders  
- Uses **angles of joints** for improved performance  

---

## Requirements
Install required dependencies:
pip install opencv-python mediapipe tensorflow scikit-learn numpy

text

---

## Usage

Set mode in the script before running:
MODE = "collect" # "collect", "train", or "predict"
MP_MODE = "hands" # "hands" or "holistic"

text

### 1. Data Collection
Collect gesture data for each action:
python main.py

text
Controls:
- **Space** → Pause/Resume  
- **n** → Next sequence  
- **s** → Skip to next action  
- **q** → Quit  

Data is saved in:
MP_Data/<action>/<sequence>/<frame>.npy

text

### 2. Train the Model
Train the LSTM model:
python main.py

text
- Trained model is saved as **`action.h5`**
- TensorBoard logs saved in `Logs/`

### 3. Predict in Real-Time
Predict signs live from webcam:
python main.py

text
- Displays predictions on screen  
- Saves all predictions to a list  

---

## Project Structure
├── main.py # Main script for all modes
├── MP_Data/ # Collected dataset
├── action.h5 # Saved trained model
├── Logs/ # TensorBoard logs

text

---

## How It Works
1. **Mediapipe** extracts landmarks from hands, pose, and face  
2. **Keypoints & angles** used as feature vectors per frame  
3. **Sequences of 30 frames** represent one gesture  
4. **LSTM neural network** learns temporal dependencies  
5. Predictions made in real-time  

---

## Customize Your Gestures
Modify the `actions` list in the config section:
actions = np.array(['hello', 'indian', 'again', 'sign', 'man', 'woman'])

text

To add new gestures:
1. Add names to `actions` list  
2. Collect new data (`MODE="collect"`)  
3. Retrain the model (`MODE="train"`)  

---

## Future Improvements
- Expand vocabulary of signs  
- Use attention-based deep learning models  
- Export as **TensorFlow Lite** for mobile use  
- Add **Text-to-Speech** for accessibility  

---

## Acknowledgements
- [Mediapipe](https://github.com/google/mediapipe) – Landmark detection  
- [TensorFlow/Keras](https://www.tensorflow.org/) – Deep learning framework  
- [OpenCV](https://opencv.org/) – Real-time video processing  
