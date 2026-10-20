import csv
from datetime import datetime, time
from pathlib import Path


class AttendanceSession:
    """Track detections and generate a final class attendance report."""

    START_TIME = time(9, 30)
    END_TIME = time(10, 0)

    FIELDNAMES = [
        "student_id",
        "student_name",
        "date",
        "timestamp",
        "status",
        "emotion",
    ]

    def __init__(self, students, csv_path="attendance/session_report.csv"):
        self.students = {
            str(student["student_id"]).strip(): {
                "student_id": str(student["student_id"]).strip(),
                "student_name": str(student["student_name"]).strip(),
            }
            for student in students
        }

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.detections = {}

    def record_detection(self, student_id, emotion, timestamp=None):
        """Record the first valid identification of an enrolled student."""
        timestamp = timestamp or datetime.now()

        if not (self.START_TIME <= timestamp.time() < self.END_TIME):
            return False, "Detection is outside the attendance window."

        student_id = str(student_id).strip()

        if student_id not in self.students:
            return False, "Student is not in the class roster."

        if student_id not in self.detections:
            self.detections[student_id] = {
                "timestamp": timestamp,
                "emotion": str(emotion).strip(),
            }

        return True, "Student detection recorded."

    def finalize(self, timestamp=None):
        """Generate the final report after the attendance window closes."""
        timestamp = timestamp or datetime.now()

        if timestamp.time() < self.END_TIME:
            return False, "Finalize attendance after 10:00 AM."

        with self.csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

            for student_id, student in self.students.items():
                detection = self.detections.get(student_id)
                present = detection is not None

                writer.writerow({
                    "student_id": student_id,
                    "student_name": student["student_name"],
                    "date": timestamp.strftime("%Y-%m-%d"),
                    "timestamp": (
                        detection["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        if present else ""
                    ),
                    "status": "Present" if present else "Absent",
                    "emotion": detection["emotion"] if present else "",
                })

        return True, f"Final attendance report saved to {self.csv_path}."