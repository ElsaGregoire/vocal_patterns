from vocal_patterns.ml_logic.model import train_model
from vocal_patterns.ml_logic.preprocessor import (
    load_file,
    mel_spectrogram,
    process_audio,
)


def preprocess(params):
    """Preprocesses the data"""

    # The processing functions from ml_logic
    load_file(audio_path)
    process_audio(y, sr)
    mel_spectrogram(y_trunc, sr)
    y = `something`
    X_preprocessed = 'something'
    pass


# @mlflow_run
def train(params):
    # The training function from ml_logic
    train_model(model, data)
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
