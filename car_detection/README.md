\# Car Colour Detection and Traffic Monitoring System



\## Overview

A computer vision application that detects vehicles and people in traffic images using YOLO object detection. It estimates vehicle colours and displays vehicle and pedestrian counts.



\## Features

\- Detects cars using YOLO.

\- Estimates the dominant colour of each detected car.

\- Draws red bounding boxes around blue cars.

\- Draws blue bounding boxes around cars of other colours.

\- Counts detected cars and people.

\- Displays the processed image through a Tkinter GUI.

\- Allows users to upload traffic images for analysis.



\## Technologies Used

\- Python

\- YOLO (Ultralytics)

\- OpenCV

\- PyTorch

\- NumPy

\- Pillow

\- Tkinter



\## Execution

From the repository root:



```bash

pip install -r car\_detection/requirements.txt

python car\_detection/app.py

