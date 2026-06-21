import argparse
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from src.gender_detector import config


def load_vectors(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    return X, y


def build_candidates():
    return {"logistic_regression": LogisticRegression(max_iter=1000), "perceptron": Perceptron(), "svc": CalibratedClassifierCV(SVC(C=10), ensemble=False)}


def train_and_select():
    X_train, y_train = load_vectors(config.TRAIN_VECTORS)
    X_test, y_test = load_vectors(config.TEST_VECTORS)

    best_name, best_model, best_score = None, None, -1.0

    for name, model in build_candidates().items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name:>20}: {score * 100:.2f}%")

        if score > best_score:
            best_name, best_model, best_score = name, model, score

    print(f"\nBest model: {best_name} ({best_score * 100:.2f}%)")
    return best_model, best_name, best_score


def main():
    parser = argparse.ArgumentParser(description="Train the gender classifier from CelebA embeddings.")
    parser.add_argument("--output", default=str(config.MODEL_PATH), help="Where to save the trained model.")
    args = parser.parse_args()

    model, name, score = train_and_select()

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved {name} to {args.output}")


if __name__ == "__main__":
    main()
