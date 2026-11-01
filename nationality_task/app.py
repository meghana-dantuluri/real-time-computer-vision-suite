import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from deepface import DeepFace
import os

class NationalityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nationality & Appearance Analysis")
        self.root.geometry("1000x700")
        self.root.configure(bg="#eef2f7")

        self.image_path = None
        self.preview_image = None

        tk.Label(
            root, text="Nationality & Appearance Analysis",
            font=("Arial", 22, "bold"), bg="#eef2f7", fg="#172554"
        ).pack(pady=15)

        tk.Label(
            root,
            text="Select a category manually. Nationality is not inferred from facial appearance.",
            font=("Arial", 10), bg="#eef2f7", fg="#475569"
        ).pack(pady=3)

        main = tk.Frame(root, bg="#eef2f7")
        main.pack(fill="both", expand=True, padx=25, pady=15)

        left = tk.Frame(main, bg="white", padx=15, pady=15)
        left.pack(side="left", fill="both", expand=True, padx=10)

        right = tk.Frame(main, bg="white", padx=20, pady=15)
        right.pack(side="right", fill="both", expand=True, padx=10)

        tk.Label(left, text="Image Preview", font=("Arial", 15, "bold"),
                 bg="white").pack(pady=10)

        self.image_label = tk.Label(
            left, text="Upload an image to preview",
            bg="#e2e8f0", width=48, height=20
        )
        self.image_label.pack(pady=10)

        tk.Button(
            left, text="Upload Image", command=self.upload_image,
            bg="#2563eb", fg="white", font=("Arial", 12, "bold"),
            padx=15, pady=8
        ).pack(pady=15)

        tk.Label(right, text="Analysis Settings", font=("Arial", 15, "bold"),
                 bg="white").pack(pady=10)

        tk.Label(right, text="Select nationality category:",
                 bg="white", font=("Arial", 11)).pack(anchor="w", pady=8)

        self.category = tk.StringVar(value="Indian")
        self.dropdown = ttk.Combobox(
            right, textvariable=self.category,
            values=["Indian", "United States", "African", "Other"],
            state="readonly", font=("Arial", 11)
        )
        self.dropdown.pack(fill="x", pady=5)

        tk.Button(
            right, text="Analyze Image", command=self.analyze,
            bg="#16a34a", fg="white", font=("Arial", 12, "bold"),
            padx=15, pady=8
        ).pack(pady=15)

        tk.Label(right, text="Analysis Results", font=("Arial", 15, "bold"),
                 bg="white").pack(pady=10)

        self.results = tk.Text(
            right, height=18, width=48, font=("Arial", 11),
            wrap="word", bg="#f8fafc"
        )
        self.results.pack(fill="both", expand=True)

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return

        self.image_path = path
        image = Image.open(path).convert("RGB")
        image.thumbnail((430, 350))
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image, text="")

        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, "Image uploaded successfully.\n")

    def get_clothing_colour(self):
        image = cv2.imread(self.image_path)
        if image is None:
            return "Unavailable"

        h, w = image.shape[:2]

        # Approximate clothing region: lower half of the image.
        crop = image[int(h * 0.55):int(h * 0.90), int(w * 0.20):int(w * 0.80)]
        if crop.size == 0:
            return "Unavailable"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        pixels = pixels[pixels[:, 2] > 35]

        if len(pixels) == 0:
            return "Black or very dark"

        hue = float(np.median(pixels[:, 0]))
        saturation = float(np.median(pixels[:, 1]))
        value = float(np.median(pixels[:, 2]))

        if value < 65:
            return "Black / dark"
        if saturation < 35:
            return "White / grey"
        if hue < 10 or hue >= 170:
            return "Red"
        if hue < 25:
            return "Orange / brown"
        if hue < 35:
            return "Yellow"
        if hue < 85:
            return "Green"
        if hue < 130:
            return "Blue"
        if hue < 155:
            return "Purple"
        return "Pink"

    def analyze(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please upload an image first.")
            return

        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, "Analyzing image... Please wait.\n")
        self.root.update()

        category = self.category.get()

        try:
            analysis = DeepFace.analyze(
                img_path=self.image_path,
                actions=["age", "emotion"],
                enforce_detection=True,
                detector_backend="opencv"
            )

            if isinstance(analysis, list):
                analysis = analysis[0]

            age = round(analysis["age"])
            emotion = analysis["dominant_emotion"]

            output = [
                "ANALYSIS COMPLETE",
                "------------------------------",
                f"Selected category: {category}",
                f"Estimated dominant emotion: {emotion}"
            ]

            if category in ["Indian", "United States"]:
                output.append(f"Estimated age: {age} years")

            if category in ["Indian", "African"]:
                colour = self.get_clothing_colour()
                output.append(f"Estimated clothing colour: {colour}")

            if category == "Other":
                output.append("Nationality category: Other")

            output.extend([
                "",
                "NOTE:",
                "Age and emotion are model estimates.",
                "Clothing colour is estimated from an approximate image region.",
                "Nationality category was selected by the user, not inferred from the face."
            ])

            self.results.delete("1.0", tk.END)
            self.results.insert(tk.END, "\n".join(output))

        except Exception as e:
            self.results.delete("1.0", tk.END)
            self.results.insert(
                tk.END,
                "Analysis could not be completed.\n\n"
                "Please use a clear image containing a visible face.\n\n"
                f"Details: {str(e)}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = NationalityApp(root)
    root.mainloop()