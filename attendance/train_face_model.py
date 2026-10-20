import cv2
import json
from pathlib import Path

DATASET_DIR = Path("datasets/student_faces")
MODEL_DIR = Path("attendance")
CASCADE_PATH = Path(
    r"C:\Users\megha\Emotion_detection\haarcascade_frontalface_default.xml"
)

MODEL_PATH = MODEL_DIR / "student_lbph.yml"
LABELS_PATH = MODEL_DIR / "student_labels.json"


def train_model():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is unavailable. Install opencv-contrib-python."
        )

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise FileNotFoundError(f"Cannot load cascade: {CASCADE_PATH}")

    faces = []
    labels = []
    label_map = {}
    label_number = 0

    for student_dir in sorted(DATASET_DIR.iterdir()):
        if not student_dir.is_dir():
            continue

        student_faces = []

        for image_path in sorted(student_dir.glob("*")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            detected = cascade.detectMultiScale(
                image, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )

            if len(detected) == 0:
                # Captured images are already cropped face samples.
                face = cv2.resize(image, (200, 200))
            else:
                x, y, w, h = max(detected, key=lambda box: box[2] * box[3])
                face = cv2.resize(image[y:y+h, x:x+w], (200, 200))

            student_faces.append(face)

        if student_faces:
            label_map[str(label_number)] = student_dir.name
            faces.extend(student_faces)
            labels.extend([label_number] * len(student_faces))
            print(f"Student {student_dir.name}: {len(student_faces)} samples")
            label_number += 1

    if not faces:
        raise RuntimeError("No usable face images found. Capture samples first.")

    if len(label_map) < 2:
        raise RuntimeError(
            "At least 2 enrolled students are required for this training setup. "
            "Capture consented samples for another student, then run again."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, __import__("numpy").array(labels, dtype="int32"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))

    with LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=4)

    print("\nTraining completed successfully!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Student labels saved to: {LABELS_PATH}")


if __name__ == "__main__":
    train_model()