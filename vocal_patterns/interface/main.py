import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from vocal_patterns.ml_logic.data import get_data
from vocal_patterns.ml_logic.model import compile_model, init_model, fit_model, evaluate_model
from vocal_patterns.ml_logic.preprocessor import (
    preprocess_audio,
)
from vocal_patterns.ml_logic.registry import load_model, save_model, save_results
from tensorflow.keras.utils import to_categorical

# @mlflow_run
def train(
    learning_rate=0.001, batch_size=32, patience=2, split_ratio: float = 0.2
) -> float:
    data = get_data()

    X = data.drop(columns=["exercise", "technique", "filename"])
    y = data[["exercise"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train_preprocessed = preprocess_audio(X_train)

    X_train_reshaped = X_train_preprocessed.reshape(
        len(X_train_preprocessed),
        X_train_preprocessed.shape[1],
        X_train_preprocessed.shape[2],
        1,
    )

    X_test_preprocessed = preprocess_audio(X_test)
    X_test_reshaped = X_test_preprocessed.reshape(
        len(X_test_preprocessed),
        X_test_preprocessed.shape[1],
        X_test_preprocessed.shape[2],
        1,
    )

    num_classes = 3

    label_encoder = LabelEncoder()
    y_train_labels = label_encoder.fit_transform(np.ravel(y_train, order="c"))
    y_train_cat = to_categorical(y_train_labels, num_classes=num_classes)

    y_test_labels = label_encoder.transform(np.ravel(y_test, order="c"))
    y_test_cat = to_categorical(y_test_labels, num_classes=num_classes)

    # model = load_model()
    # if model is None:
    model = init_model(input_shape=X_train_reshaped.shape[1:])

    model = compile_model(model=model, learning_rate=learning_rate)

    model, history = fit_model(
        model,
        X_train_reshaped,
        y_train_cat,
        batch_size,
        patience,
        validation_split=split_ratio,
    )

    # Evaluate the model on the test data using `evaluate`
    loss, accuracy = model.evaluate(X_test_reshaped, y_test_cat)

    params = dict(
        context="train",
        loss=loss,
        row_count=len(X_train_reshaped),
    )

    save_results(params=params, metrics=dict(accuracy=accuracy))
    save_model(model=model)

    print(accuracy)
    return model


def predict(X_pred: pd.DataFrame = None):
    if X_pred is None:
        raise ValueError("No data to predict on!")

    model = load_model()
    assert model is not None

    X_pred_processed = preprocess_audio(X_pred)
    X_pred_reshaped = X_pred_processed.reshape(
        len(X_pred_processed),
        X_pred_processed.shape[1],
        X_pred_processed.shape[2],
        1,
    )

    y_pred = model.predict(X_pred_reshaped)

    return y_pred
