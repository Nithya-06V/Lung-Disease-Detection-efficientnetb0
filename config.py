import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")
DB_NAME = os.getenv("DB_NAME", "lung_disease_db")

MODEL_NAME = os.getenv("MODEL_NAME", "EfficientNetB0")
MODEL_FILES = {
    "EfficientNetB0": "efficientnetb0.keras",
    "DenseNet121": "densenet121.keras",
    "ResNet50": "resnet50.keras",
}
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "model" / MODEL_FILES.get(MODEL_NAME, "lung_disease_model.keras")))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", str(BASE_DIR / "model" / "class_names.json"))
METRICS_PATH = os.getenv("METRICS_PATH", str(BASE_DIR / "model" / f"{MODEL_NAME.lower()}_metrics.json"))
COMPARISON_PATH = os.getenv("COMPARISON_PATH", str(BASE_DIR / "model" / "model_comparison.json"))
IMAGE_SIZE = (224, 224)
CLASS_NAMES = []

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY", "lung-disease-ai-secret-key")


def get_db_config():
    return {
        "host": DB_HOST,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": DB_NAME,
        "autocommit": True,
    }
