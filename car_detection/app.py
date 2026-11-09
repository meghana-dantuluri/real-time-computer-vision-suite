import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "yolo11n.pt")
model = YOLO(MODEL_PATH)

class CarDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Colour Detection & Traffic Analysis")
        self.root.geometry("1200x800")
        self.root.configure(bg="#20242c")

        tk.Label(
            root, text="CAR COLOUR DETECTION SYSTEM",
            font=("Arial", 22, "bold"), fg="white", bg="#20242c"
        ).pack(pady=15)

        tk.Button(
            root, text="Upload Traffic Image", command=self.upload_image,
            font=("Arial", 13, "bold"), bg="#2196F3", fg="white",
            padx=15, pady=8
        ).pack(pady=10)

        self.preview = tk.Label(root, bg="#303640")
        self.preview.pack(pady=10, expand=True)

        self.result_label = tk.Label(
            root, text="Upload an image to begin analysis",
            font=("Arial", 15, "bold"), fg="white", bg="#20242c"
        )
        self.result_label.pack(pady=15)

        self.image_ref = None

    def get_colour(self, crop):
        if crop.size == 0:
            return "Unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Ignore very dark, bright and low-saturation pixels.
        mask = (v > 35) & (v < 245) & (s > 35)

        if np.count_nonzero(mask) < 30:
            return "Black/White/Grey"

        hue = h[mask]
        saturation = s[mask]
        value = v[mask]

        # Use the most common hue range.
        if np.mean(value) < 65:
            return "Black"
        if np.mean(saturation) < 55:
            return "White/Grey"

        hist = np.bincount(hue, minlength=180)
        dominant_hue = int(np.argmax(hist))

        if dominant_hue <= 10 or dominant_hue >= 170:
            return "Red"
        elif dominant_hue <= 22:
            return "Orange"
        elif dominant_hue <= 35:
            return "Yellow"
        elif dominant_hue <= 85:
            return "Green"
        elif dominant_hue <= 130:
            return "Blue"
        elif dominant_hue <= 160:
            return "Purple"
        else:
            return "Red"

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not open the selected image.")
            return

        try:
            results = model(image, verbose=False)
            cars = 0
            people = 0
            blue_cars = 0
            other_cars = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if confidence < 0.35:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1], x2)
                    y2 = min(image.shape[0], y2)

                    if class_id == 2:  # COCO car class
                        cars += 1
                        crop = image[y1:y2, x1:x2]
                        colour = self.get_colour(crop)

                        if colour == "Blue":
                            box_colour = (0, 0, 255)  # Red rectangle
                            blue_cars += 1
                        else:
                            box_colour = (255, 0, 0)  # Blue rectangle
                            other_cars += 1

                        label = f"{colour} Car {confidence:.2f}"

                    elif class_id == 0:  # COCO person class
                        people += 1
                        box_colour = (0, 255, 0)
                        label = f"Person {confidence:.2f}"

                    else:
                        continue

                    cv2.rectangle(
                        image, (x1, y1), (x2, y2), box_colour, 2
                    )
                    cv2.putText(
                        image, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        box_colour, 2
                    )

            summary = (
                f"Cars detected: {cars}   |   "
                f"Blue cars: {blue_cars}   |   "
                f"Other cars: {other_cars}   |   "
                f"People detected: {people}"
            )

            self.result_label.config(text=summary)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            pil_image.thumbnail((1050, 560))

            self.image_ref = ImageTk.PhotoImage(pil_image)
            self.preview.config(image=self.image_ref)

        except Exception as e:
            messagebox.showerror("Detection Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = CarDetectionApp(root)
    root.mainloop()