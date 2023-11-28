import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from vocal_patterns.ml_logic.data import get_data
from vocal_patterns.ml_logic.model import compile_model, init_model, fit_model
from vocal_patterns.ml_logic.preprocessor import (
    preprocess_audio,
)
from vocal_patterns.ml_logic.registry import load_model, save_model, save_results


# @mlflow_run
def train(
    learning_rate=0.0005, batch_size=256, patience=2, split_ratio: float = 0.1
) -> float:
    data = get_data()
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.3)
    X_train_preprocessed = preprocess_audio(X_train)

    model = load_model()

    if model is None:
        model = init_model(input_shape=X_train_preprocessed.shape[1:])

    model = compile_model(model=model, learning_rate=learning_rate)

    model = fit_model(
        model,
        X_train_preprocessed,
        y_train,
        batch_size,
        patience,
        validation_split=split_ratio,
    )

    # Evaluate the model on the test data using `evaluate`
    loss, accuracy = model.evaluate(X_test, y_test)

    params = dict(
        context="train",
        loss=loss,
        row_count=len(X_train_preprocessed),
    )

    save_results(params=params, metrics=dict(accuracy=accuracy))
    save_model(model=model)

    return accuracy


def predict(X_pred: pd.DataFrame = None):
    if X_pred is None:
        raise ValueError("No data to predict on!")

    model = load_model()
    assert model is not None

    X_pred_processed = preprocess_audio(X_pred)
    y_pred = model.predict(X_pred_processed)

    return y_pred
