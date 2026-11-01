\# Nationality Category and Appearance Analysis



\## Overview

A Python GUI application that accepts an uploaded image and displays estimated age, dominant emotion and clothing colour according to a user-selected nationality category.



\## Features

\- Image upload and preview

\- User-selectable nationality category

\- Facial emotion analysis using DeepFace

\- Estimated age analysis

\- Approximate clothing colour detection using OpenCV

\- Conditional output based on selected category



\## Category-wise Output



| Category | Output |

|---|---|

| Indian | Age, emotion, clothing colour |

| United States | Age, emotion |

| African | Emotion, clothing colour |

| Other | Selected category, emotion |



\## Technologies

\- Python

\- Tkinter

\- OpenCV

\- DeepFace

\- TensorFlow

\- Pillow

\- NumPy



\## Run Instructions



Activate the project's Python virtual environment and install dependencies:



```bash

pip install -r nationality\_task/requirements.txt

python nationality\_task/app.py

