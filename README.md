Real-Time Computer Vision Suite

A modular computer vision and machine learning project developed as part of a Data Science / Machine Learning internship. The repository is organized into separate modules for the assigned computer vision tasks.

Current Progress

Attendance System

The attendance module combines face recognition, emotion detection, attendance logging, and session reporting.

Implemented features:
- Trains an LBPH face-recognition model using collected student face samples.
- Recognizes enrolled student IDs through a webcam.
- Detects facial expressions using an emotion classification model.
- Records student ID, name, date, timestamp, attendance status, and predicted emotion.
- Restricts official attendance recording to **9:30 AM–10:00 AM**.
- Prevents duplicate attendance records for the same student on the same date.
- Marks enrolled students who were not detected as absent in the session report.
- Includes automated unit tests for attendance logging and session management.

Limitations: Recognition accuracy depends on lighting, camera quality, face samples, and the configured confidence threshold. Emotion predictions are model estimates and may be inaccurate. Test mode produces a separate test report and should not be treated as official attendance.

Technology Stack

- Python 3.11
- OpenCV and OpenCV-contrib (LBPH face recognition)
- TensorFlow / Keras (emotion classification)
- NumPy
- Pillow
- scikit-learn
- Python `unittest`

Setup

Clone the repository and open a terminal in its root directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bat
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Use Python 3.11 for the current dependency setup.

Attendance System Setup

1. Prepare the student face dataset

Collect face samples for each enrolled student using `attendance/capture_faces.py`. Keep the collected images in the local `datasets/student_faces/` directory, organized by student ID.

Use the required consent and privacy precautions when collecting or processing anyone's face images.

2. Train the face-recognition model

From the repository root, run:

```bash
python attendance/train_face_model.py
```

The training process generates the LBPH model and student-label mapping locally. These generated files are excluded from Git because they may contain or reveal information about enrolled students.

3. Prepare the student roster

Create a local `attendance/students.json` file containing the student IDs and corresponding names in the format expected by `attendance/run_attendance.py`.

Do not commit the real roster to the public repository.

4. Prepare the emotion model files

The emotion detector loads a Keras model architecture, model weights, and a Haar cascade XML file. Place the required files in the locations configured in `attendance/run_attendance.py`, or update those paths for your own environment.

The original emotion-model files are not included in this repository. Confirm the relevant licence or obtain permission before redistributing third-party model files.

5. Run the attendance system

From the repository root, run:

```bash
python attendance/run_attendance.py
```

Official attendance recording is limited to **9:30 AM–10:00 AM according to the computer's local clock**. Use test mode only for testing:

```bash
python attendance/run_attendance.py --test-mode
```

Test mode writes a separate test report. Check the script's console instructions for how to exit the webcam session.

Run the Tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover attendance logging and session-management behavior. They do not guarantee the real-world accuracy of face recognition or emotion classification.

Project Structure

- `attendance/` — face recognition, emotion detection, attendance logging, and session management
- `animal_detection/` — planned animal detection module
- `car_detection/` — planned vehicle detection module
- `drowsiness_detection/` — planned drowsiness detection module
- `nationality_task/` — awaiting technical clarification; nationality should not be inferred from facial appearance
- `sign_language/` — planned sign-language recognition module
- `common/` — shared utilities
- `tests/` — automated tests
- `docs/` — project documentation
- `datasets/` — local dataset workspace; data files are excluded from Git

Data Privacy

Student face images, student rosters, trained face-recognition files, and generated attendance reports are excluded from the public repository. Share datasets only with appropriate authorization and consent, using access-controlled storage where necessary.

Dataset sources, permissions, and any approved Google Drive submission link will be documented once finalized.

Status

Development is ongoing. Refer to the source code and tests for the current implementation. Planned modules are not represented as completed features.
