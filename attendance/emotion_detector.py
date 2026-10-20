from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

EMOTIONS_LIST = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


class EmotionDetector:
    """Predict facial expressions using the existing emotion model."""

    def __init__(self, model_json, model_weights, cascade_path):
        model_json = Path(model_json)
        model_weights = Path(model_weights)
        cascade_path = Path(cascade_path)

        for path in (model_json, model_weights, cascade_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required emotion model file not found: {path}"
                )

        with model_json.open("r", encoding="utf-8") as file:
            self.model = model_from_json(file.read())

        self.model.load_weights(str(model_weights))

        self.face_detector = cv2.CascadeClassifier(str(cascade_path))

        if self.face_detector.empty():
            raise ValueError(
                "Could not load the face detection cascade."
            )

    def predict_face(self, face_gray):
        """Predict an emotion from one grayscale face crop."""

        if face_gray is None or face_gray.size == 0:
            raise ValueError("The face image is empty.")

        face = cv2.resize(face_gray, (48, 48))
        face = face.astype(np.float32)

        prediction = self.model.predict(
            face[np.newaxis, :, :, np.newaxis],
            verbose=0,
        )

        emotion_index = int(np.argmax(prediction))
        return EMOTIONS_LIST[emotion_index]

    def predict_image(self, image_path):
        """Detect faces and predict their emotions in an image."""

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
        )

        results = []

        for (x, y, width, height) in faces:
            face_gray = gray[y:y + height, x:x + width]
            emotion = self.predict_face(face_gray)

            results.append({
                "box": (
                    int(x),
                    int(y),
                    int(width),
                    int(height),
                ),
                "emotion": emotion,
            })

        return results