
from pathlib import Path
from ultralytics import YOLO

# Path to your trained model
MODEL_PATH = Path("models/best.pt")

# Path to an image or a folder containing images
SOURCE_PATH = Path("sample_images")

# Folder where prediction results will be saved
OUTPUT_PATH = Path("predictions")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Place best.pt inside the models folder."
        )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {SOURCE_PATH}. "
            "Add an image or folder of images."
        )

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=str(SOURCE_PATH),
        save=True,
        project=str(OUTPUT_PATH),
        name="ppe_results",
        exist_ok=True,
        conf=0.25
    )

    print(f"Predictions saved in {OUTPUT_PATH}/ppe_results")

if __name__ == "__main__":
    main()
