import json
import os
import subprocess
import sys
import tempfile


def classify_image(image_path, country="IND"):
    """Classify an image with SpeciesNet and return detected species."""
    image_path = os.path.abspath(image_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(
            temp_dir, "speciesnet_predictions.json"
        )

        command = [
            sys.executable,
            "-m",
            "speciesnet.scripts.run_model",
            "--filepaths",
            image_path,
            "--predictions_json",
            output_path,
            "--country",
            country,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "SpeciesNet failed:\n" + result.stderr[-3000:]
            )

        with open(output_path, "r", encoding="utf-8") as file:
            data = json.load(file)

    predictions = data.get("predictions", [])
    if not predictions:
        return []

    animals = []

    for detection in predictions[0].get("detections", []):
        classification = detection.get(
            "processed_classification", {}
        )

        scientific_path = classification.get("class", "")
        score = classification.get("score", 0.0)

        species = (
            scientific_path.split(";")[-1].strip()
            if scientific_path else "Unknown animal"
        )

        if detection.get("label") == "animal":
            animals.append({
                "species": species or "Unknown animal",
                "confidence": float(score),
                "bbox": detection.get("bbox", []),
            })

    return animals