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
from keras import optimizers, layers, models


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    num_classes = 3
    y_train_cat = target_encoder(y_train, num_classes=num_classes)
    y_test_cat = target_encoder(y_test, num_classes=num_classes)

    model = init_model(input_shape=X_train.shape[1:])

    model = compile_model(model=model, learning_rate=learning_rate)

    model, history = fit_model(
        model,
        X_train,
        y_train_cat,
        batch_size,
        patience,
        validation_split=split_ratio,
    )

    # Evaluate the model on the test data using `evaluate`
    loss, accuracy = model.evaluate(X_test, y_test_cat)

    results_params = dict(
        context="train",
        learning_rate=learning_rate,
        data_split=split_ratio,
        data_augmentations=augmentations,
        loss=loss,
        row_count=len(X_train),
    )

    save_results(params=results_params, metrics=dict(accuracy=accuracy))
    save_model(model=model)

    print(accuracy)
    return model


def predict(X_pred_processed: np.ndarray = None):
    if X_pred_processed is None:
        raise ValueError("No data to predict on!")

    # model = load_model()
    model = models.load_model("/prod/mlops/training_outputs/models/20231121-160757.h5")
    assert model is not None

    y_pred = model.predict(X_pred_processed)
    print(y_pred)

    return y_pred


if __name__ == "__main__":
    train()
