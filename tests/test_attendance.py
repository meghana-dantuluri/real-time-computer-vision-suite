import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from attendance.attendance import AttendanceLogger


class TestAttendanceLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        csv_path = Path(self.temp_dir.name) / "attendance.csv"
        self.logger = AttendanceLogger(csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_records_present_student(self):
        timestamp = datetime(2026, 10, 13, 9, 45)
        success, message = self.logger.mark_present(
            "ST001", "Student One", timestamp
        )
        self.assertTrue(success, message)

        with self.logger.csv_path.open(
            newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["student_id"], "ST001")
        self.assertEqual(rows[0]["status"], "Present")

    def test_rejects_outside_attendance_window(self):
        timestamp = datetime(2026, 10, 13, 10, 15)
        success, _ = self.logger.mark_present(
            "ST002", "Student Two", timestamp
        )
        self.assertFalse(success)

    def test_prevents_duplicate_attendance(self):
        timestamp = datetime(2026, 10, 13, 9, 45)
        self.logger.mark_present("ST003", "Student Three", timestamp)

        success, _ = self.logger.mark_present(
            "ST003", "Student Three", timestamp
        )
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()