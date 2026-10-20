import cv2
from pathlib import Path

DATASET_DIR = Path("datasets/student_faces")
CASCADE_PATH = Path("C:/Users/megha/Emotion_detection/haarcascade_frontalface_default.xml")
SAMPLES_REQUIRED = 40


def capture_faces(student_id):
    save_dir = DATASET_DIR / student_id
    save_dir.mkdir(parents=True, exist_ok=True)

    face_detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_detector.empty():
        raise FileNotFoundError("Could not load the Haar cascade file.")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    count = 0
    print("Look at the camera and slowly turn your face slightly.")
    print("Press Q to stop capturing.")

    try:
        while count < SAMPLES_REQUIRED:
            success, frame = camera.read()
            if not success:
                print("Could not read from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))

                if count < SAMPLES_REQUIRED:
                    count += 1
                    cv2.imwrite(str(save_dir / f"face_{count:03d}.jpg"), face)

            cv2.putText(
                frame, f"Samples: {count}/{SAMPLES_REQUIRED}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.imshow("Student Face Enrollment", frame)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {count} face samples in: {save_dir.resolve()}")


if __name__ == "__main__":
    student_id = input("Enter student ID: ").strip()
    if not student_id or any(c in student_id for c in r'\/:*?"<>|'):
        raise ValueError("Enter a valid student ID.")
    capture_faces(student_id)