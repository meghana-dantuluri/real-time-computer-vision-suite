# **Real-Time Computer Vision Suite**



A modular computer vision and machine learning project being developed as part of a Data Science / Machine Learning internship. The project extends an existing emotion detection application and develops separate modules for the six assigned tasks.



#### **Current Progress**



###### **Attendance System — Initial Module**

\- Records student ID, name, date, timestamp, and attendance status in CSV format.

\- Restricts attendance recording to 9:30–10:00 AM.

\- Prevents duplicate attendance records for the same student on the same date.

\- Includes automated unit tests for the recording logic.



**Current limitation**: Student identification, automatic absence marking, and GUI integration have not yet been implemented.



###### **Setup**



Python 3.11 is recommended for the current development environment.



```bash

python -m venv .venv

```



Activate the environment on Windows:



```bat

.venv\\Scripts\\activate

```



Install dependencies:



```bash

python -m pip install -r requirements.txt

```



###### **Run the Attendance Tests**



From the repository root:



```bash

python -m unittest tests.test\_attendance -v

```



###### **Project Modules**



\- `attendance/` — attendance recording

\- `animal\_detection/` — planned animal detection module

\- `drowsiness\_detection/` — planned drowsiness detection module

\- `nationality\_task/` — task awaiting technical clarification

\- `sign\_language/` — planned sign-language recognition module

\- `car\_detection/` — planned vehicle detection module

\- `common/` — shared utilities

\- `tests/` — automated tests

\- `docs/` — project documentation

\- `datasets/` — local dataset workspace; dataset files are not automatically published



###### **Dataset Access**



Dataset sources, usage permissions, and the Google Drive dataset link will be documented here once finalized.



###### **Status**



Development is ongoing. Only features explicitly described as implemented have been tested so far.

