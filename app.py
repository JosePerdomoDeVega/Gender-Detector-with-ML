import tempfile
import gradio as gr
from PIL import Image
from src.gender_detector import GenderPredictor

predictor = GenderPredictor()


def detect(image: Image.Image) -> str:
    if image is None:
        return "Please upload an image or take a photo first."

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        prediction = predictor.predict(tmp.name)

    if prediction.confidence is not None:
        return f"Predicted: {prediction.label} ({prediction.confidence * 100:.1f}% confidence)"
    return f"Predicted: {prediction.label}"


def build_interface() -> gr.Interface:
    return gr.Interface(
        fn=detect,
        inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Image"),
        outputs=gr.Text(label="Result"),
        title="Gender Detector",
        description=(
            "Upload a photo or take one with your webcam. The model crops the face, turns it into "
            "a Facenet512 embedding and classifies the apparent gender. Predictions reflect "
            "patterns in the CelebA training data and should be read as such."
        ),
        flagging_mode="never",
    )


if __name__ == "__main__":
    build_interface().launch()
