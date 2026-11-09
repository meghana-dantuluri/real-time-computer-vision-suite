import os
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "gesture_data.csv")
MODEL_FILE = os.path.join(BASE, "gesture_model.pkl")

GESTURES = ["Hello", "Stop", "Yes", "No", "Thank You"]

# These are custom static-gesture labels for this prototype,
# not a complete or certified sign-language vocabulary.


class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("1350x850")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#eef2ff")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

        self.cap = None
        self.camera_running = False
        self.collecting = False
        self.sample_count = 0
        self.frame_count = 0
        self.model = None
        self.photo = None

        self.create_gui()
        self.load_model()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def operating_hours(self):
        now = datetime.now()
        return 18 <= now.hour < 22

    def create_gui(self):
        tk.Label(
            self.root, text="SIGN LANGUAGE RECOGNITION",
            font=("Arial", 24, "bold"),
            bg="#eef2ff", fg="#172554"
        ).pack(pady=12)

        tk.Label(
            self.root,
            text="Gesture training | Image prediction | Live recognition",
            font=("Arial", 11), bg="#eef2ff", fg="#475569"
        ).pack(pady=4)

        main = tk.Frame(self.root, bg="#eef2ff")
        main.pack(fill="both", expand=True, padx=18, pady=12)

        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Preview panel
        left = tk.Frame(main, bg="white", padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=8)

        tk.Label(
            left, text="Camera / Image Preview",
            font=("Arial", 16, "bold"), bg="white"
        ).pack(pady=8)

        self.preview_label = tk.Label(
            left, text="Camera preview will appear here",
            bg="#e2e8f0", fg="#334155",
            font=("Arial", 13), width=65, height=27
        )
        self.preview_label.pack(fill="both", expand=True, padx=5, pady=8)

        tk.Button(
            left, text="Start Live Recognition",
            command=self.start_camera,
            bg="#2563eb", fg="white",
            font=("Arial", 11, "bold"), pady=9
        ).pack(fill="x", padx=35, pady=5)

        tk.Button(
            left, text="Stop Camera",
            command=self.stop_camera,
            bg="#dc2626", fg="white",
            font=("Arial", 11, "bold"), pady=9
        ).pack(fill="x", padx=35, pady=5)

        # Controls panel
        right = tk.Frame(main, bg="white", padx=18, pady=14)
        right.grid(row=0, column=1, sticky="nsew", padx=8)

        tk.Label(
            right, text="Model Training",
            font=("Arial", 17, "bold"), bg="white"
        ).pack(pady=10)

        tk.Label(
            right, text="Select gesture label:",
            bg="white", font=("Arial", 11)
        ).pack(anchor="w", pady=5)

        self.gesture = tk.StringVar(value=GESTURES[0])
        ttk.Combobox(
            right, textvariable=self.gesture,
            values=GESTURES, state="readonly",
            font=("Arial", 11)
        ).pack(fill="x", pady=6)

        tk.Label(
            right,
            text="Collect 100 samples for each gesture.\n"
                 "Vary your hand position slightly while collecting.",
            bg="white", fg="#64748b",
            font=("Arial", 10), justify="center"
        ).pack(pady=8)

        self.collect_button = tk.Button(
            right, text="Collect Training Samples",
            command=self.collect_samples,
            bg="#7c3aed", fg="white",
            font=("Arial", 11, "bold"), pady=11
        )
        self.collect_button.pack(fill="x", pady=7)

        tk.Button(
            right, text="Train ML Model",
            command=self.train_model,
            bg="#16a34a", fg="white",
            font=("Arial", 11, "bold"), pady=11
        ).pack(fill="x", pady=7)

        tk.Button(
            right, text="Upload Image & Predict",
            command=self.upload_image,
            bg="#0891b2", fg="white",
            font=("Arial", 11, "bold"), pady=11
        ).pack(fill="x", pady=7)

        tk.Label(
            right, text="Prediction / Status",
            font=("Arial", 16, "bold"), bg="white"
        ).pack(pady=15)

        self.result = tk.Label(
            right, text="Waiting for input...",
            font=("Arial", 13, "bold"),
            fg="#1d4ed8", bg="#f1f5f9",
            wraplength=350, height=5
        )
        self.result.pack(fill="x", pady=6)

        self.status = tk.Label(
            right, text="Ready",
            bg="white", fg="#64748b"
        )
        self.status.pack(pady=6)

        tk.Label(
            right,
            text="Operating hours: 6 PM – 10 PM IST",
            bg="white", fg="#64748b",
            font=("Arial", 9)
        ).pack(pady=8)

    def extract_features(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, result

        hand = result.multi_hand_landmarks[0]
        points = np.array(
            [[p.x, p.y, p.z] for p in hand.landmark],
            dtype=np.float32
        )

        points -= points[0]
        scale = np.max(np.linalg.norm(points[:, :2], axis=1))

        if scale > 0:
            points /= scale

        return points.flatten(), result

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((640, 480))

        canvas = Image.new("RGB", (640, 480), "#e2e8f0")
        x = (640 - image.width) // 2
        y = (480 - image.height) // 2
        canvas.paste(image, (x, y))

        self.photo = ImageTk.PhotoImage(canvas)
        self.preview_label.config(image=self.photo, text="")

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror("Camera Error", "Cannot access webcam.")
            return False
        return True

    def collect_samples(self):
        if self.collecting or self.camera_running:
            messagebox.showwarning("Camera Busy", "Stop the current operation first.")
            return

        if not self.open_camera():
            return

        self.collecting = True
        self.sample_count = 0
        self.frame_count = 0
        self.status.config(text=f"Collecting: {self.gesture.get()}")
        self.collect_frame()

    def collect_frame(self):
        if not self.collecting:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.collect_frame)
            return

        frame = cv2.flip(frame, 1)
        features, result = self.extract_features(frame)

        if result.multi_hand_landmarks:
            self.drawer.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )

        self.frame_count += 1

        # Capture every third frame to reduce near-duplicate samples.
        if features is not None and self.frame_count % 3 == 0:
            new_file = not os.path.exists(DATA_FILE)

            with open(DATA_FILE, "a", newline="") as file:
                writer = csv.writer(file)
                if new_file:
                    writer.writerow(["label"] + [f"f{i}" for i in range(63)])
                writer.writerow([self.gesture.get()] + features.tolist())

            self.sample_count += 1

        cv2.putText(
            frame, f"{self.gesture.get()}: {self.sample_count}/100",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        self.show_frame(frame)

        if self.sample_count >= 100:
            self.collecting = False
            self.release_camera()
            self.status.config(text="Collection complete")
            messagebox.showinfo(
                "Samples Collected",
                f"100 samples collected for {self.gesture.get()}."
            )
            return

        self.root.after(30, self.collect_frame)

    def train_model(self):
        if not os.path.exists(DATA_FILE):
            messagebox.showwarning("No Dataset", "Collect gesture samples first.")
            return

        labels, features = [], []

        with open(DATA_FILE, "r", newline="") as file:
            rows = list(csv.reader(file))

        for row in rows[1:]:
            if len(row) == 64:
                labels.append(row[0])
                features.append([float(v) for v in row[1:]])

        missing = set(GESTURES) - set(labels)
        if missing:
            messagebox.showwarning(
                "Missing Gestures",
                "Collect samples for: " + ", ".join(sorted(missing))
            )
            return

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2,
            random_state=42, stratify=labels
        )

        model = RandomForestClassifier(
            n_estimators=150, random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        joblib.dump(model, MODEL_FILE)
        self.model = model

        self.status.config(text="Model trained successfully")
        messagebox.showinfo(
            "Training Complete",
            f"Model trained successfully!\n"
            f"Test accuracy: {accuracy * 100:.2f}%\n"
            f"Training samples: {len(X_train)}\n"
            f"Testing samples: {len(X_test)}"
        )

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.status.config(text="Saved model loaded")
            except Exception:
                self.model = None

    def predict(self, frame):
        if not self.operating_hours():
            messagebox.showwarning(
                "Outside Operating Hours",
                "Recognition is available only from 6 PM to 10 PM IST."
            )
            return

        if self.model is None:
            messagebox.showwarning(
                "Model Not Trained",
                "Collect all five gesture classes and train the model first."
            )
            return

        features, result = self.extract_features(frame)

        if features is None:
            self.result.config(text="No hand detected.")
            return

        prediction = self.model.predict([features])[0]
        confidence = max(self.model.predict_proba([features])[0]) * 100

        self.result.config(
            text=f"Recognized word: {prediction}\n"
                 f"Confidence: {confidence:.1f}%"
        )

    def upload_image(self):
        if not self.operating_hours():
            messagebox.showwarning(
                "Outside Operating Hours",
                "Image recognition is available only from 6 PM to 10 PM IST."
            )
            return

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Image Error", "Cannot read selected image.")
            return

        self.show_frame(frame)
        self.predict(frame)

    def start_camera(self):
        if not self.operating_hours():
            messagebox.showwarning(
                "Outside Operating Hours",
                "Live recognition is available only from 6 PM to 10 PM IST."
            )
            return

        if self.model is None:
            messagebox.showwarning(
                "Model Not Trained", "Train the model first."
            )
            return

        if self.camera_running or self.collecting:
            return

        if not self.open_camera():
            return

        self.camera_running = True
        self.status.config(text="Live recognition active")
        self.update_camera()

    def update_camera(self):
        if not self.camera_running:
            return

        if not self.operating_hours():
            self.stop_camera()
            messagebox.showwarning(
                "Operating Hours Ended",
                "Recognition has stopped. Available from 6 PM to 10 PM IST."
            )
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.update_camera)
            return

        frame = cv2.flip(frame, 1)
        features, result = self.extract_features(frame)

        if result.multi_hand_landmarks:
            self.drawer.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )

        if features is not None:
            prediction = self.model.predict([features])[0]
            confidence = max(self.model.predict_proba([features])[0]) * 100

            cv2.putText(
                frame, f"{prediction} ({confidence:.0f}%)",
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2
            )
            self.result.config(
                text=f"Recognized word: {prediction}\n"
                     f"Confidence: {confidence:.1f}%"
            )

        self.show_frame(frame)
        self.root.after(30, self.update_camera)

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop_camera(self):
        self.collecting = False
        self.camera_running = False
        self.release_camera()
        self.status.config(text="Camera stopped")

    def close(self):
        self.stop_camera()
        self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()