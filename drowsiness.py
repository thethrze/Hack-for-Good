import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from collections import deque
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_connections = mp.solutions.face_mesh_connections
LEFT_EYE = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE = [362, 385, 387, 263, 380, 373, 390, 249]
MOUTH = [61, 291, 0, 17, 13, 14, 312, 308]
def eye_aspect_ratio(landmarks, indices, frame_w, frame_h):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in indices]
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)
def mouth_aspect_ratio(landmarks, indices, frame_w, frame_h):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in indices]
    # Vertical distance (top lip to bottom lip)
    A = np.linalg.norm(np.array(points[4]) - np.array(points[5]))
    # Horizontal distance (mouth corners)
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return A / C
features = []
labels = []
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Press 'a' for Awake, 'd' for Drowsy while collecting frames.")
while len(features) < 200:  # collect 200 labeled frames
    ret, frame = cap.read()
    if not ret:
        continue
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            mouth_ratio = mouth_aspect_ratio(face_landmarks.landmark, MOUTH, w, h)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1)
            if key == ord('a'):  # Awake
                features.append([avg_ear, mouth_ratio])
                labels.append(0)
            elif key == ord('d'):  # Drowsy
                features.append([avg_ear, mouth_ratio])
                labels.append(1)

cap.release()
cv2.destroyAllWindows()
X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
joblib.dump(clf, "drowsiness_model.pkl")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
prediction_window = deque(maxlen=5)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    text = "No face detected"
    color = (0, 255, 255)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2
            mouth_ratio = mouth_aspect_ratio(face_landmarks.landmark, MOUTH, w, h)
            prediction = clf.predict([[avg_ear, mouth_ratio]])[0]
            prediction_window.append(prediction)
            if np.mean(prediction_window) > 0.5:
                text = "DROWSY!"
                color = (0, 0, 255)
            else:
                text = "Awake"
                color = (0, 255, 0)
            mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_connections.FACEMESH_TESSELATION)
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.imshow("Drowsiness ML", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()