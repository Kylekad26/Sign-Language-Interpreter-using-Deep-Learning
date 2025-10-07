# serve_model.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and setup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "action.h5")
model = tf.keras.models.load_model(MODEL_PATH)

actions = np.array(['hello', 'indian', 'again', 'sign', 'man', 'woman', 'you', 'hearing', 'teacher', 'thank you', 'welcome',
     'sorry','practice', 'good', 'bad','thin'])
sequence_length = 30

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

MP_MODE = "hands"  # Use hands or holistic, same as Start.py


def calc_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(np.dot(v1, v2), -1, 1))


def hand_angles(hand_landmarks):
    if hand_landmarks is None:
        return np.zeros(21 * 21)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    angles = []
    for i in range(21):
        for j in range(21):
            if i != j:
                angles.append(calc_angle(pts[i], pts[j]))
            else:
                angles.append(0.0)
    return np.array(angles)


def extract_keypoints(results):
    if MP_MODE == "holistic":
        pose = np.array([[res.x, res.y, res.z, res.visibility]
                         for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z]
                         for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z]
                       for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(
            21 * 3)
        rh = np.array([[res.x, res.y, res.z]
                       for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(
            21 * 3)
        lh_angles = hand_angles(results.left_hand_landmarks)
        rh_angles = hand_angles(results.right_hand_landmarks)
        return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), lh_angles, rh.flatten(), rh_angles])
    else:
        if results.multi_hand_landmarks:
            all_hands = results.multi_hand_landmarks
            lh = np.array([[lm.x, lm.y, lm.z] for lm in all_hands[0].landmark]) if len(all_hands) > 0 else np.zeros(
                (21, 3))
            rh = np.array([[lm.x, lm.y, lm.z] for lm in all_hands[1].landmark]) if len(all_hands) > 1 else np.zeros(
                (21, 3))
            lh_angles = hand_angles(all_hands[0]) if len(all_hands) > 0 else np.zeros(21 * 21)
            rh_angles = hand_angles(all_hands[1]) if len(all_hands) > 1 else np.zeros(21 * 21)
        else:
            lh = np.zeros((21, 3));
            rh = np.zeros((21, 3))
            lh_angles = np.zeros(21 * 21);
            rh_angles = np.zeros(21 * 21)
        return np.concatenate([lh.flatten(), lh_angles, rh.flatten(), rh_angles])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# This endpoint receives a sequence of image frames
@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    # Load mediapipe model
    if MP_MODE == "holistic":
        mp_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        mp_model = mp_hands.Hands(static_image_mode=False,
                                  max_num_hands=2,
                                  min_detection_confidence=0.45,
                                  min_tracking_confidence=0.45)

    sequence = []
    for file in files[:sequence_length]:  # Only use up to sequence_length frames
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        _, results = mediapipe_detection(img, mp_model)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

    if len(sequence) != sequence_length:
        return {"error": "Need {} frames, received {}".format(sequence_length, len(sequence))}

    x = np.expand_dims(np.array(sequence), axis=0)
    res = model.predict(x)[0]
    word = str(actions[np.argmax(res)])
    confidence = float(res[np.argmax(res)])

    return {"predicted_word": word, "confidence": confidence}
