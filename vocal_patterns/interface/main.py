import pandas as pd
from vocal_patterns.ml_logic.model import train_model
from vocal_patterns.ml_logic.preprocessor import (
    load_file,
    mel_spectrogram,
    process_audio,
)


# @mlflow_run
def train(X: pd.DataFrame, y: pd.DataFrame):
    X_preprocessed = "#preprocess X"

    train_model(X_preprocessed, y)
    pass


# @mlflow_run
def evaluate(params):
    pass


def predict(params):
    pass


if __name__ == "__main__":
    preprocess(something)
    train(something)
    evaluate(something)
    predict(something)
