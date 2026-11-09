# Real-Time Computer Vision Suite

A modular computer vision and machine learning project developed as part of a Data Science / Machine Learning internship.

## Current Progress

### Attendance System

The attendance module combines face recognition, emotion detection, attendance logging, and session reporting.

**Implemented features**

- Trains an LBPH face-recognition model using student face samples.
- Recognizes enrolled student IDs through a webcam.
- Detects facial expressions using an emotion classification model.
- Records student ID, name, date, timestamp, attendance status, and predicted emotion in CSV format.
- Restricts official attendance recording to **9:30 AM–10:00 AM**.
- Prevents duplicate attendance records for the same student on the same date.
- Marks enrolled students who were not detected as absent in the session report.
- Includes automated unit tests for attendance logging and session management.

**Limitations:** Recognition accuracy depends on lighting, camera quality, face samples, and the configured confidence threshold. Emotion predictions may be inaccurate. Test mode creates a separate report and should not be treated as official attendance.

## Technology Stack

- Python 3.11
- OpenCV-contrib
- TensorFlow / Keras
- NumPy
- Pillow
- scikit-learn
- Python `unittest`

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bat
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

## Running the Attendance System

1. Collect student face samples using `attendance/capture_faces.py`.
2. Train the face-recognition model:

   ```bash
   python attendance/train_face_model.py
   ```

3. Prepare your local `attendance/students.json` roster in the format expected by the application.
4. Make the required emotion-model architecture, weights, and Haar cascade files available at the paths configured in the code.
5. Start the application:

   ```bash
   python attendance/run_attendance.py
   ```

For testing outside the official attendance workflow:

```bash
python attendance/run_attendance.py --test-mode
```

Official attendance is restricted to 9:30 AM–10:00 AM according to the computer's local clock.

**Note:** The original emotion-model files are not included in this repository. Confirm licensing or obtain permission before redistributing them.

## Project Structure

- `attendance/` — face recognition, emotion detection, attendance logging, and session management
- `animal_detection/` — planned animal detection module
- `car_detection/` — planned vehicle detection module
- `drowsiness_detection/` — planned drowsiness detection module
- `nationality_task/` — awaiting technical clarification
- `sign_language/` — planned sign-language recognition module
- `common/` — shared utilities
- `tests/` — automated tests
- `docs/` — project documentation
- `datasets/` — local dataset workspace

## Data Privacy

Personal or sensitive data, including student face images, student rosters, identifiable attendance records, and private datasets, should not be committed to the public repository.

Data collection and processing must be performed only with appropriate authorization and consent. Any datasets containing personal information should be stored in access-controlled locations.

The sign language module includes a gesture-landmark dataset and a trained classification model for the supported static gestures. These are separate from personal face images and attendance records.

Dataset sources, usage permissions, and submission links should be documented and verified before final submission.

## Status

All six assigned project modules have been implemented:

1. Emotion Detection
2. Animal Detection
3. Drowsiness Detection
4. Nationality Category and Appearance Analysis
5. Sign Language Detection
6. Car Colour Detection and Traffic Monitoring

Development of the assigned modules is complete. Final testing, documentation review, and submission preparation are in progress.

The applications are prototypes and have the limitations described in their respective module documentation.
