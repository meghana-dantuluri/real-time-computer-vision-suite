
import argparse
import csv
import json
import traceback
from datetime import datetime, time
from pathlib import Path

import cv2

from attendance.emotion_detector import EmotionDetector

ROOT = Path(__file__).resolve().parent.parent
ATTENDANCE_DIR = ROOT / "attendance"

MODEL_PATH = ATTENDANCE_DIR / "student_lbph.yml"
LABELS_PATH = ATTENDANCE_DIR / "student_labels.json"
ROSTER_PATH = ATTENDANCE_DIR / "students.json"

ORIGINAL_DIR = Path(r"C:\Users\megha\Emotion_detection")
CASCADE_PATH = ORIGINAL_DIR / "haarcascade_frontalface_default.xml"
EMOTION_JSON = ORIGINAL_DIR / "model_a1.json"
EMOTION_WEIGHTS = ORIGINAL_DIR / "model_weights1.h5"

START_TIME = time(9, 30)
END_TIME = time(10, 0)
CONFIDENCE_THRESHOLD = 65


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test the webcam without recording official attendance."
    )
    args = parser.parse_args()

    required_files = [
        MODEL_PATH,
        LABELS_PATH,
        ROSTER_PATH,
        CASCADE_PATH,
        EMOTION_JSON,
        EMOTION_WEIGHTS,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    now = datetime.now()

    if not args.test_mode and not (
        START_TIME <= now.time() < END_TIME
    ):
        print("Official attendance works only from 9:30 AM to 10:00 AM.")
        print("Use --test-mode to test without recording official attendance.")
        return

    with LABELS_PATH.open("r", encoding="utf-8") as file:
        labels = json.load(file)

    with ROSTER_PATH.open("r", encoding="utf-8") as file:
        roster_data = json.load(file)

    roster = {
        str(student["student_id"]).strip(): {
            "student_name": str(student["student_name"]).strip()
        }
        for student in roster_data
    }

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        raise RuntimeError("Could not load the face-detection cascade.")

    emotion_detector = EmotionDetector(
        EMOTION_JSON,
        EMOTION_WEIGHTS,
        CASCADE_PATH,
    )

    detections = {}
    today = now.strftime("%Y-%m-%d")

    report_name = (
        "attendance_test_report.csv"
        if args.test_mode
        else "attendance_report.csv"
    )
    report_path = ATTENDANCE_DIR / report_name

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    if args.test_mode:
        print("TEST MODE: this is not official attendance.")
    else:
        print("OFFICIAL ATTENDANCE MODE.")

    print("Press Q to finish.")

    try:
        while args.test_mode or datetime.now().time() < END_TIME:
            ok, frame = camera.read()

            if not ok:
                print("Could not read webcam frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80),
            )

            for (x, y, w, h) in faces:
                face_gray = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face_gray, (200, 200))

                label, confidence = recognizer.predict(resized_face)
                student_id = labels.get(str(label))

                if (
                    confidence > CONFIDENCE_THRESHOLD
                    or student_id not in roster
                ):
                    display = "Unknown"
                    box_color = (0, 0, 255)

                else:
                    student = roster[student_id]
                    display = (
                        f"{student['student_name']} ({student_id})"
                    )
                    box_color = (0, 255, 0)

                    if student_id not in detections:
                        try:
                            emotion = emotion_detector.predict_face(
                                face_gray
                            )
                        except Exception:
                            print(
                                f"\nEmotion prediction failed for "
                                f"student {student_id}:"
                            )
                            traceback.print_exc()
                            emotion = "Unavailable"

                        detections[student_id] = {
                            "student_id": student_id,
                            "student_name": student["student_name"],
                            "date": today,
                            "timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "status": "Present",
                            "emotion": emotion,
                        }

                        print(
                            f"Recognized: {display} | "
                            f"Emotion: {emotion}"
                        )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2,
                )

                cv2.putText(
                    frame,
                    display,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

            cv2.putText(
                frame,
                f"Recognized: {len(detections)} | Press Q to finish",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Student Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    fields = [
        "student_id",
        "student_name",
        "date",
        "timestamp",
        "status",
        "emotion",
    ]

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for student_id, student in roster.items():
            row = detections.get(
                student_id,
                {
                    "student_id": student_id,
                    "student_name": student["student_name"],
                    "date": today,
                    "timestamp": "",
                    "status": "Absent",
                    "emotion": "",
                },
            )
            writer.writerow(row)

    print(f"\nReport saved to: {report_path}")
    print(f"Present: {len(detections)}")
    print(f"Absent: {len(roster) - len(detections)}")


if __name__ == "__main__":
    main()
