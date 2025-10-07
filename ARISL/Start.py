import cv2
import numpy as np
import os
import mediapipe as mp
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import math
import time

# ================================ CONFIGURATION ================================
MODE = "predict"  # "collect", "train", or "predict"
MP_MODE = "hands" # "hands" or "holistic"
actions = np.array(
    ['hello', 'indian', 'again', 'sign', 'man', 'woman', 'you', 'hearing', 'teacher', 'thank you', 'welcome',
     'sorry','practice', 'good', 'bad','thin'])
no_sequences = 60
sequence_length = 30
DATA_PATH = os.path.join('MP_Data')

OVERWRITE = False  # Set True to enable overwriting data if it already exists

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calc_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(np.dot(v1, v2), -1, 1))

def hand_angles(hand_landmarks):
    if hand_landmarks is None:
        return np.zeros(21*21)
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    angles = []
    for i in range(21):
        for j in range(21):
            if i != j: angles.append(calc_angle(pts[i], pts[j]))
            else: angles.append(0.0)
    return np.array(angles)

def extract_keypoints(results):
    if MP_MODE == "holistic":
        pose = np.array([[res.x, res.y, res.z, res.visibility]
                         for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z]
                         for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z]
                        for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z]
                        for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros(21*3)
        lh_angles = hand_angles(results.left_hand_landmarks)
        rh_angles = hand_angles(results.right_hand_landmarks)
        return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), lh_angles, rh.flatten(), rh_angles])
    else:
        if results.multi_hand_landmarks:
            all_hands = results.multi_hand_landmarks
            lh = np.array([[lm.x, lm.y, lm.z] for lm in all_hands[0].landmark]) if len(all_hands) > 0 else np.zeros((21,3))
            rh = np.array([[lm.x, lm.y, lm.z] for lm in all_hands[1].landmark]) if len(all_hands) > 1 else np.zeros((21,3))
            lh_angles = hand_angles(all_hands[0]) if len(all_hands) > 0 else np.zeros(21*21)
            rh_angles = hand_angles(all_hands[1]) if len(all_hands) > 1 else np.zeros(21*21)
        else:
            lh = np.zeros((21,3)); rh = np.zeros((21,3))
            lh_angles = np.zeros(21*21); rh_angles = np.zeros(21*21)
        return np.concatenate([lh.flatten(), lh_angles, rh.flatten(), rh_angles])

def draw_landmarks(image, results):
    if MP_MODE == "holistic":
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    else:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# ================================ DATA COLLECTION ================================
if MODE == "collect":
    for action in actions:
        for sequence in range(no_sequences):
            seq_dir = os.path.join(DATA_PATH, action, str(sequence))
            if OVERWRITE and os.path.exists(seq_dir):
                shutil.rmtree(seq_dir)
                print(f"Overwriting data: removed {seq_dir}")
            try:
                os.makedirs(seq_dir)
            except:
                pass
    cap = cv2.VideoCapture(0)
    paused = False
    current_action = 0
    current_sequence = 0
    current_frame = 0

    if MP_MODE == "holistic":
        mp_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        mp_model = mp_hands.Hands(static_image_mode=False,
                                  max_num_hands=2,
                                  min_detection_confidence=0.45,
                                  min_tracking_confidence=0.45)

    with mp_model as model:
        while current_action < len(actions):
            action = actions[current_action]
            if current_sequence >= no_sequences:
                current_action += 1
                current_sequence = 0
                current_frame = 0
                continue

            seq_dir = os.path.join(DATA_PATH, action, str(current_sequence))
            if not OVERWRITE and os.path.exists(seq_dir) and len(os.listdir(seq_dir)) >= sequence_length:
                print(f"Data exists for {action} seq {current_sequence+1}, skipping. Set OVERWRITE=True to overwrite.")
                current_sequence += 1
                current_frame = 0
                continue

            ret, frame = cap.read()
            if not ret: break
            image, results = mediapipe_detection(frame, model)
            draw_landmarks(image, results)
            status_text = f'Action: {action} | Sequence: {current_sequence+1}/{no_sequences} | Frame: {current_frame+1}/{sequence_length}'
            cv2.putText(image, status_text, (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            if paused:
                cv2.putText(image, 'PAUSED - Press SPACE to continue', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
            else:
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(current_sequence), str(current_frame))
                np.save(npy_path, keypoints)
                current_frame += 1
                if current_frame >= sequence_length:
                    current_sequence += 1
                    current_frame = 0
                    print(f"Completed sequence {current_sequence} for {action}")
            cv2.imshow('OpenCV Feed', image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): paused = not paused
            elif key == ord('n') and not paused:
                current_sequence += 1
                current_frame = 0
            elif key == ord('s') and not paused:
                current_action += 1
                current_sequence = 0
                current_frame = 0
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection completed!")

# ================================ TRAINING ================================
if MODE == "train":
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    sample_action = actions[0]
    sample_file = os.path.join(DATA_PATH, sample_action, "0", "0.npy")
    keypoint_length = np.load(sample_file).shape[0]
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, keypoint_length)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001)
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, early_stop, reduce_lr], validation_data=(X_test, y_test))
    res = model.predict(X_test)
    yhat = np.argmax(res, axis=1).tolist()
    ytrue = np.argmax(y_test, axis=1).tolist()
    print(f"Test accuracy: {accuracy_score(ytrue, yhat)}")
    model.save('action.h5')
    print("Model saved as 'action.h5'")

# ================================ PREDICTION ================================
if MODE == "predict":
    model = tf.keras.models.load_model('action.h5')
    threshold = 0.5
    cap = cv2.VideoCapture(0)
    all_predictions = []

    if MP_MODE == "holistic":
        mp_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        mp_model = mp_hands.Hands(static_image_mode=False,
                                  max_num_hands=2,
                                  min_detection_confidence=0.45,
                                  min_tracking_confidence=0.45)

    prompt_duration = 2  # seconds

    with mp_model as model_mp:
        while cap.isOpened():
            start = time.time()
            while time.time() - start < prompt_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                overlay = frame.copy()
                alpha = 0.6
                cv2.rectangle(overlay, (0,0), (640, 60), (0,128,255), -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, "Do the action now!", (30,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Predicted words:")
                    print(all_predictions)
                    exit()

            sequence = []
            predicted = ""
            for _ in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, model_mp)
                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                cv2.imshow('OpenCV Feed', image)
                k = cv2.waitKey(10) & 0xFF
                if k == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Predicted words:")
                    print(all_predictions)
                    exit()

            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                word = actions[np.argmax(res)]
                conf = res[np.argmax(res)]
                if conf > threshold:
                    predicted = word
                    all_predictions.append(predicted)

                display_start = time.time()
                while time.time() - display_start < 1:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = mediapipe_detection(frame, model_mp)
                    draw_landmarks(image, results)
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                    cv2.putText(image, f'Predicted: {predicted}', (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Predicted words:")
                        print(all_predictions)
                        exit()

        cap.release()
        cv2.destroyAllWindows()
        print("Predicted words:")
        print(all_predictions)