from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Images-CelebA-1K"
MODELS_DIR = ROOT_DIR / "models"

TRAIN_VECTORS = DATA_DIR / "train_df_vectors.csv"
TEST_VECTORS = DATA_DIR / "test_df_vectors.csv"

MODEL_PATH = MODELS_DIR / "gender_classifier.joblib"

EMBEDDING_MODEL = "Facenet512"
DETECTOR_BACKEND = "opencv"
