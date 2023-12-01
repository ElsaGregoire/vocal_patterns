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
from vocal_patterns.ml_logic.registry import load_model, save_model, save_results


# @mlflow_run
def train(
    learning_rate=0.001,
    batch_size=32,
    patience=2,
    split_ratio: float = 0.2,
    augmentations: list | None = None,
) -> float:
    data = get_data()

    try:
        data = pd.read_pickle("preproc.pkl")
        print("Loaded cached preprocessing data")
    except FileNotFoundError:
        print("No cached preprocessing data found, preprocessing now...")
        data = preprocess_df(data)
        data.to_pickle("preproc.pkl")

    X = data.drop(
        columns=[
            "exercise",
            "technique",
        ]
    )
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
    save_model(model=model)

    print(accuracy)
    return model


def predict(X_pred_processed: np.ndarray = None):
    if X_pred_processed is None:
        raise ValueError("No data to predict on!")

    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred_processed)
    print(y_pred)

    return y_pred


if __name__ == "__main__":
    train()
