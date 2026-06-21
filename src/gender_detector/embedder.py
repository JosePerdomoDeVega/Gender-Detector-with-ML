import numpy as np
from src.gender_detector import config


def get_face_embedding(image_path: str) -> np.ndarray:
    from deepface import DeepFace

    embedding_objs = DeepFace.represent(
        img_path=image_path, model_name=config.EMBEDDING_MODEL, detector_backend=config.DETECTOR_BACKEND, enforce_detection=False)

    return np.array(embedding_objs[0]["embedding"])
