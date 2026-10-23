import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from animal_detection.species_classifier import classify_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# Simplified project categories, not a universal biological diet classification.
CARNIVORE_CLASSES = {"cat", "bear"}

# Animal classes available in the pretrained COCO model.
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
}


class AnimalDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection System")
        self.root.geometry("1100x760")
        self.root.configure(bg="#f2f4f8")

        self.model = YOLO("yolo11n.pt")
        self.video_capture = None
        self.video_running = False
        self.carnivore_total = 0
        self.image_reference = None

        title = tk.Label(
            root,
            text="Animal Detection System",
            font=("Arial", 22, "bold"),
            bg="#f2f4f8",
            fg="#1d3557"
        )
        title.pack(pady=12)

        subtitle = tk.Label(
            root,
            text="Detect animals in images and videos",
            font=("Arial", 11),
            bg="#f2f4f8"
        )
        subtitle.pack(pady=(0, 10))

        controls = tk.Frame(root, bg="#f2f4f8")
        controls.pack(pady=8)

        tk.Button(
            controls, text="Select Image",
            command=self.select_image,
            width=16, bg="#457b9d", fg="white"
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            controls, text="Select Video",
            command=self.select_video,
            width=16, bg="#457b9d", fg="white"
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            controls, text="Stop Video",
            command=self.stop_video,
            width=16, bg="#e09f3e", fg="black"
        ).pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(
            root, text="Choose an image or video to begin.",
            font=("Arial", 11), bg="#f2f4f8"
        )
        self.status.pack(pady=8)

        self.preview = tk.Label(
            root, bg="white",
            text="Detection preview will appear here",
            font=("Arial", 14),
            width=90, height=28
        )
        self.preview.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def draw_detections(self, frame):
        results = self.model.predict(
            source=frame, imgsz=320, conf=0.35, verbose=False
        )

        annotated = frame.copy()
        animal_count = 0
        carnivore_count = 0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id].lower()

                if class_name not in ANIMAL_CLASSES:
                    continue

                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                animal_count += 1
                is_carnivore = class_name in CARNIVORE_CLASSES

                # Red for configured carnivore categories, green for others.
                color = (0, 0, 255) if is_carnivore else (0, 200, 0)

                if is_carnivore:
                    carnivore_count += 1

                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2), color, 2
                )
                cv2.putText(
                    annotated, label,
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA
                )

        return annotated, animal_count, carnivore_count

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        max_width, max_height = 1000, 530
        image.thumbnail((max_width, max_height))

        self.image_reference = ImageTk.PhotoImage(image)
        self.preview.configure(
            image=self.image_reference, text=""
        )


    def select_image(self):
        self.stop_video()

        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror(
                "Error", "The selected image could not be opened."
            )
            return

        self.status.configure(text="Classifying image with SpeciesNet...")
        self.root.update_idletasks()

        try:
            detections = classify_image(path, country="IND")
        except Exception as error:
            messagebox.showerror(
                "SpeciesNet Error",
                f"SpeciesNet could not classify this image.\n\n{error}"
            )
            self.status.configure(text="Image classification failed.")
            return

        annotated = frame.copy()
        height, width = annotated.shape[:2]
        animal_count = 0
        carnivore_count = 0

        for detection in detections:
            species = detection["species"].lower()
            confidence = detection["confidence"]
            bbox = detection.get("bbox", [])

            if len(bbox) != 4:
                continue

            # SpeciesNet bbox format: normalized x, y, width, height.
            x, y, box_width, box_height = bbox
            x1 = max(0, min(width - 1, int(x * width)))
            y1 = max(0, min(height - 1, int(y * height)))
            x2 = max(0, min(width - 1, int((x + box_width) * width)))
            y2 = max(0, min(height - 1, int((y + box_height) * height)))

            animal_count += 1

            # Project categories only; this is not a complete diet taxonomy.
            is_carnivore = species in {
                "lion", "tiger", "leopard", "cheetah", "snow leopard",
                "domestic cat", "cat", "bear", "wolf", "fox", "hyena"
            }

            color = (0, 0, 255) if is_carnivore else (0, 200, 0)
            if is_carnivore:
                carnivore_count += 1

            label = f"{species} {confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA
            )

        self.show_frame(annotated)
        self.status.configure(
            text=f"SpeciesNet | Animals: {animal_count} | "
                 f"Configured carnivores: {carnivore_count}"
        )

        messagebox.showinfo(
            "Detection Results",
            f"SpeciesNet image analysis complete.\n"
            f"Animals detected: {animal_count}\n"
            f"Configured carnivores detected: {carnivore_count}"
        )


    def select_video(self):
        self.stop_video()

        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.video_capture = cv2.VideoCapture(path)

        if not self.video_capture.isOpened():
            self.video_capture = None
            messagebox.showerror(
                "Error", "The selected video could not be opened."
            )
            return

        self.carnivore_total = 0
        self.video_running = True
        self.status.configure(text="Processing video...")
        self.process_video_frame()

    def process_video_frame(self):
        if not self.video_running or self.video_capture is None:
            return

        success, frame = self.video_capture.read()

        if not success:
            total = self.carnivore_total
            self.stop_video()
            self.status.configure(
                text=f"Video finished | Configured carnivores: {total}"
            )
            messagebox.showinfo(
                "Video Results",
                f"Video processing complete.\n"
                f"Total configured carnivore detections across frames: {total}"
            )
            return

        annotated, animal_count, carnivore_count = (
            self.draw_detections(frame)
        )

        self.carnivore_total += carnivore_count
        self.show_frame(annotated)

        self.status.configure(
            text=f"Animals in current frame: {animal_count} | "
                 f"Configured carnivores in frame: {carnivore_count} | "
                 f"Running frame total: {self.carnivore_total}"
        )

        self.root.after(1, self.process_video_frame)

    def stop_video(self):
        self.video_running = False

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def close_app(self):
        self.stop_video()
        self.root.destroy()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    root = tk.Tk()
    app = AnimalDetectionGUI(root)
    root.mainloop()