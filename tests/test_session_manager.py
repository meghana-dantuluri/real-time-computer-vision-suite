import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from attendance.session_manager import AttendanceSession


class TestAttendanceSession(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "report.csv"
        self.students = [
            {"student_id": "S001", "student_name": "Student One"},
            {"student_id": "S002", "student_name": "Student Two"},
        ]
        self.session = AttendanceSession(
            self.students, csv_path=self.csv_path
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_records_identified_student(self):
        moment = datetime(2026, 10, 20, 9, 40)
        ok, _ = self.session.record_detection("S001", "Happy", moment)
        self.assertTrue(ok)

    def test_rejects_detection_outside_window(self):
        moment = datetime(2026, 10, 20, 10, 5)
        ok, _ = self.session.record_detection("S001", "Neutral", moment)
        self.assertFalse(ok)

    def test_final_report_marks_present_and_absent(self):
        seen = datetime(2026, 10, 20, 9, 40)
        finished = datetime(2026, 10, 20, 10, 1)

        self.session.record_detection("S001", "Happy", seen)
        ok, _ = self.session.finalize(finished)
        self.assertTrue(ok)

        with self.csv_path.open(newline="", encoding="utf-8") as file:
            import csv
            rows = {
                row["student_id"]: row
                for row in csv.DictReader(file)
            }

        self.assertEqual(rows["S001"]["status"], "Present")
        self.assertEqual(rows["S001"]["emotion"], "Happy")
        self.assertEqual(rows["S002"]["status"], "Absent")

    def test_cannot_finalize_before_window_ends(self):
        moment = datetime(2026, 10, 20, 9, 50)
        ok, _ = self.session.finalize(moment)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()