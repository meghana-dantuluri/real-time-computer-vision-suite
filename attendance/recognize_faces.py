import cv2
import json
from pathlib import Path

MODEL_PATH = Path("attendance/student_lbph.yml")
LABELS_PATH = Path("attendance/student_labels.json")
CASCADE_PATH = Path(
    r"C:\Users\megha\Emotion_detection\haarcascade_frontalface_default.xml"
)

# Tune this threshold using real validation images.
# Lower confidence values generally indicate closer LBPH matches.
CONFIDENCE_THRESHOLD = 65


def main():
    for path in (MODEL_PATH, LABELS_PATH, CASCADE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError("Could not load face detection cascade.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Face recognition started. Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read webcam frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in detected_faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                predicted_label, confidence = recognizer.predict(face)

                if confidence <= CONFIDENCE_THRESHOLD:
                    student_id = labels.get(str(predicted_label))
                    name = f"Student {student_id}" if student_id else "Unknown"
                else:
                    name = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{name} ({confidence:.1f})",
                    (x, max(y-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Student Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()