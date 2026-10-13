import csv
from datetime import datetime, time
from pathlib import Path


class AttendanceLogger:
    """Record student attendance in a CSV file."""

    FIELDNAMES = [
        "student_id",
        "student_name",
        "date",
        "timestamp",
        "status",
    ]

    def __init__(self, csv_path="attendance/attendance_log.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def mark_present(self, student_id, student_name, timestamp=None):
        """Record a student once per date and attendance session."""
        timestamp = timestamp or datetime.now()

        # Attendance is allowed from 9:30 AM up to 10:00 AM.
        if not (time(9, 30) <= timestamp.time() < time(10, 0)):
            return False, "Attendance is only available from 9:30 to 10:00 AM."

        student_id = str(student_id).strip()
        student_name = str(student_name).strip()

        if not student_id or not student_name:
            return False, "Student ID and name are required."

        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if (
                    row["student_id"] == student_id
                    and row["date"] == timestamp.strftime("%Y-%m-%d")
                    and row["status"] == "Present"
                ):
                    return False, "Attendance has already been recorded for this student today."

        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writerow({
                "student_id": student_id,
                "student_name": student_name,
                "date": timestamp.strftime("%Y-%m-%d"),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Present",
            })

        return True, "Attendance recorded successfully."