import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Age estimation is optional; the rest of the app works without it.
try:
    from deepface import DeepFace
except Exception as exc:
    DeepFace = None
    print("Age estimation unavailable:", exc)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Download the small pretrained person detector on first run if needed.
model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")
face_mesh = mp.solutions.face_mesh

# MediaPipe Face Mesh eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_ratio(points):
    # Eye aspect ratio: vertical eye opening relative to eye width
    a = np.linalg.norm(points[1] - points[5])
    b = np.linalg.norm(points[2] - points[4])
    c = np.linalg.norm(points[0] - points[3])
    return (a + b) / (2.0 * c) if c else 1.0

def estimate_age(face_crop):
    if DeepFace is None or face_crop.size == 0:
        return "Age unavailable"
    try:
        result = DeepFace.analyze(
            face_crop,
            actions=["age"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True
        )
        if isinstance(result, list):
            result = result[0]
        return f"Approx. age: {int(result['age'])}"
    except Exception as exc:
        print("Age estimation failed:", exc)
        return "Age unavailable"

def process_image(path):
    frame = cv2.imread(path)
    if frame is None:
        messagebox.showerror("Error", "Could not open this image.")
        return
    process_frame(frame, image_mode=True)

def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open this video.")
        return

    closed_counts = {}
    alert_shown = False
    frame_no = 0

    with face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=20,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_no += 1
            results = model(frame, classes=[0], verbose=False)
            people = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                people.append((x1, y1, x2, y2))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = mesh.process(rgb).multi_face_landmarks or []
            face_data = []

            h, w = frame.shape[:2]
            for face in faces:
                pts = np.array(
                    [(lm.x * w, lm.y * h) for lm in face.landmark]
                )
                ear = (
                    eye_ratio(pts[LEFT_EYE]) +
                    eye_ratio(pts[RIGHT_EYE])
                ) / 2
                fx, fy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                face_data.append((fx, fy, ear, pts))

            drowsy = 0
            for idx, (x1, y1, x2, y2) in enumerate(people):
                face = next(
                    (f for f in face_data
                     if x1 <= f[0] <= x2 and y1 <= f[1] <= y2),
                    None
                )
                if face:
                    fx, fy, ear, pts = face
                    closed_counts[idx] = (
                        closed_counts.get(idx, 0) + 1 if ear < 0.19 else 0
                    )
                    is_drowsy = closed_counts[idx] >= 12
                else:
                    closed_counts[idx] = 0
                    is_drowsy = False

                color = (0, 0, 255) if is_drowsy else (0, 200, 0)
                label = "DROWSINESS ALERT" if is_drowsy else "Person"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if is_drowsy:
                    drowsy += 1

            cv2.putText(frame, f"People: {len(people)}  Alerts: {drowsy}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)
            cv2.imshow("Drowsiness Detection - press Q to stop",
                       cv2.resize(frame, (960, 600)))

            if drowsy and not alert_shown:
                alert_shown = True
                print(f"Warning: {drowsy} possible drowsy person(s).")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, image_mode=False):
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=20,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as mesh:
        results = model(frame, classes=[0], verbose=False)
        people = [
            tuple(map(int, box.xyxy[0].tolist()))
            for box in results[0].boxes
        ]
        h, w = frame.shape[:2]
        faces = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_data = []

        for face in (faces.multi_face_landmarks or []):
            pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])
            ear = (eye_ratio(pts[LEFT_EYE]) + eye_ratio(pts[RIGHT_EYE])) / 2
            fx, fy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            face_data.append((fx, fy, ear, pts))

        alerts = 0
        for x1, y1, x2, y2 in people:
            face = next((f for f in face_data
                         if x1 <= f[0] <= x2 and y1 <= f[1] <= y2), None)
            closed = bool(face and face[2] < 0.19)
            color = (0, 0, 255) if closed else (0, 200, 0)
            label = "Eyes closed - check" if closed else "Person"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if closed:
                alerts += 1
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                age = estimate_age(face_crop)
                print(f"Possible drowsiness. {age}")

        cv2.putText(frame, f"People: {len(people)}  Eyes closed: {alerts}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Image Analysis - press any key to close",
                   cv2.resize(frame, (960, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if alerts:
            messagebox.showwarning(
                "Possible Drowsiness",
                f"{alerts} person(s) appear to have closed eyes.\n"
                "This is only a visual indicator, not a diagnosis."
            )

def choose_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )
    if path:
        process_image(path)

def choose_video():
    path = filedialog.askopenfilename(
        filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")]
    )
    if path:
        process_video(path)

root = tk.Tk()
root.title("Drowsiness Detection System")
root.geometry("460x260")
root.resizable(False, False)

tk.Label(root, text="Drowsiness Detection System",
         font=("Arial", 16, "bold")).pack(pady=20)
tk.Label(root, text="Select an image or video for analysis.",
         font=("Arial", 10)).pack(pady=5)
tk.Button(root, text="Analyze Image", width=24,
          command=choose_image).pack(pady=8)
tk.Button(root, text="Analyze Video", width=24,
          command=choose_video).pack(pady=8)
tk.Label(root, text="Red = possible drowsiness / closed eyes",
         fg="red").pack(pady=8)

root.mainloop()