import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
LEFT_EYE = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE = [362, 385, 387, 263, 380, 373, 390, 249]
MOUTH = [61, 291, 0, 17, 13, 14, 312, 308]

def aspect_ratio(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def head_pose(nose, left_eye, right_eye):
    yaw = left_eye.x - right_eye.x
    pitch = nose.y - ((left_eye.y + right_eye.y) / 2)
    return yaw, pitch

def hand_near_face(hand_lms, face_lms, w, h):
    if not hand_lms:
        return 0
    face_x = int(face_lms[1].x * w)
    face_y = int(face_lms[1].y * h)
    for hand in hand_lms:
        for lm in hand.landmark:
            x, y = int(lm.x*w), int(lm.y*h)
            if abs(x - face_x) < 80 and abs(y - face_y) < 80:
                return 1
    return 0
def run_detection():
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands

    face = mp_face.FaceMesh(max_num_faces=1)
    hands = mp_hands.Hands(max_num_hands=2)

    model_path = "model.pkl"
    model_loaded = os.path.exists(model_path)

    if model_loaded:
        clf = joblib.load(model_path)
        print("[INFO] Model loaded. Running real-time detection...")
    else:
        print("[INFO] No model found. Running in DATA COLLECTION MODE...")

    features_list = []
    labels_list = []

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_res = face.process(rgb)
        hand_res = hands.process(rgb)

        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark

            ear = (aspect_ratio(lm, LEFT_EYE, w, h) +
                   aspect_ratio(lm, RIGHT_EYE, w, h)) / 2
            mar = aspect_ratio(lm, MOUTH, w, h)
            yaw, pitch = head_pose(lm[1], lm[33], lm[263])
            hand_flag = hand_near_face(hand_res.multi_hand_landmarks, lm, w, h)

            fv = [ear, mar, yaw, pitch, hand_flag]
            if model_loaded:
                pred = clf.predict([fv])[0]
                label_text = ["Awake", "Drowsy", "Look Away", "Phone"][pred]
                color = (0,255,0) if pred == 0 else (0,0,255)

                cv2.putText(frame, label_text, (20,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                cv2.putText(frame, "A=Awake | D=Drowsy | L=LookAway | P=Phone",
                            (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('a'), ord('d'), ord('l'), ord('p')]:
                    labels_map = {'a':0, 'd':1, 'l':2, 'p':3}
                    features_list.append(fv)
                    labels_list.append(labels_map[chr(key)])
                    print(f"[+] Recorded sample {chr(key).upper()}")

        cv2.imshow("Driver Attention AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if not model_loaded:
        np.save("features.npy", np.array(features_list))
        np.save("labels.npy", np.array(labels_list))
        print("[DATA SAVED] features.npy and labels.npy")
def train_model():
    if not (os.path.exists("features.npy") and os.path.exists("labels.npy")):
        print("[ERROR] No dataset found. Run detection mode first to collect data.")
        return

    X = np.load("features.npy")
    y = np.load("labels.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=300)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Model accuracy: {acc*100:.2f}%")

    joblib.dump(clf, "model.pkl")
    print("[MODEL SAVED] model.pkl")
if __name__ == "__main__":
    print("\n=== DRIVER ATTENTION AI ===")
    print("1. Collect Data / Run Detection")
    print("2. Train Model")
    choice = input("Choose an option (1/2): ")

    if choice == "1":
        run_detection()
    elif choice == "2":
        train_model()
    else:
        print("Invalid choice.")
