import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from vocal_patterns.ml_logic.data import get_data
from vocal_patterns.ml_logic.encoders import target_encoder
from vocal_patterns.ml_logic.model import (
    compile_model,
    init_model,
    fit_model,
)
from vocal_patterns.ml_logic.preprocessor import (
    preprocess_df,
)
from vocal_patterns.ml_logic.registry import save_model, save_results
from vocal_patterns.ml_logic.registry import mlflow_run


@mlflow_run
def train(
    learning_rate=0.001,
    batch_size=32,
    patience=2,
    split_ratio: float = 0.2,
) -> float:
    snippet_duration = 8
    augmentations = {
        "stretch_target_duration": snippet_duration,
        "snippets": {"duration": snippet_duration, "overlap": snippet_duration - 1},
        "background_noise": 1,
        # "noise_up": 0.001,
    }

    data = get_data()
    data = preprocess_df(data, clearCached=False, augmentations=augmentations)

    X = data["spectrogram"]
    y = data[["exercise"]]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    X_train_array = np.stack(X_train, axis=0)
    X_val_array = np.stack(X_val, axis=0)

    num_classes = 3
    y_train_cat = target_encoder(y_train, num_classes=num_classes)
    y_val_cat = target_encoder(y_val, num_classes=num_classes)

    model = init_model(input_shape=X_train_array.shape[1:])

    model = compile_model(model=model, learning_rate=learning_rate)

    model, history = fit_model(
        model,
        X_train_array,
        y_train_cat,
        batch_size,
        patience,
        validation_split=split_ratio,
    )

    # Evaluate the model on the validation data using `evaluate`
    loss, accuracy = model.evaluate(X_val_array, y_val_cat)

    results_params = dict(
        context="train",
        learning_rate=learning_rate,
        data_split=split_ratio,
        data_augmentations=augmentations,
        loss=loss,
        row_count=len(X_train_array),
    )

    save_results(params=results_params, metrics=dict(accuracy=accuracy))
    save_model(model=model, augmentations=augmentations)

    print(accuracy)
    return model


def predict(X_pred_processed: np.ndarray, model=None):
    if X_pred_processed is None:
        raise ValueError("No data to predict on!")

    assert model is not None

    y_pred = model.predict(X_pred_processed)[0]
    return y_pred


if __name__ == "__main__":
    train()
