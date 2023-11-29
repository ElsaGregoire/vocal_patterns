import numpy as np
from sklearn.model_selection import train_test_split
from vocal_patterns.ml_logic.data import get_data
from vocal_patterns.ml_logic.encoders import target_encoder
from vocal_patterns.ml_logic.model import (
    compile_model,
    init_model,
    fit_model,
)
from vocal_patterns.ml_logic.preprocessor import (
    preprocess_predict,
    preprocess_train,
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

    X = data.drop(columns=["exercise", "technique", "filename"])
    y = data[["exercise"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train_preprocessed = preprocess_train(X_train, augmentations=augmentations)
    X_test_preprocessed = preprocess_train(X_test)

    num_classes = 3
    y_train_cat = target_encoder(y_train, num_classes=num_classes)
    y_test_cat = target_encoder(y_test, num_classes=num_classes)

    model = init_model(input_shape=X_train_preprocessed.shape[1:])

    model = compile_model(model=model, learning_rate=learning_rate)

    model, history = fit_model(
        model,
        X_train_preprocessed,
        y_train_cat,
        batch_size,
        patience,
        validation_split=split_ratio,
    )

    # Evaluate the model on the test data using `evaluate`
    loss, accuracy = model.evaluate(X_test_preprocessed, y_test_cat)

    results_params = dict(
        context="train",
        learning_rate=learning_rate,
        data_split=split_ratio,
        data_augmentations=augmentations,
        loss=loss,
        row_count=len(X_train_preprocessed),
    )

    save_results(params=results_params, metrics=dict(accuracy=accuracy))
    save_model(model=model)

    print(accuracy)
    return model


def predict(X_pred: np.ndarray = None):
    if X_pred is None:
        raise ValueError("No data to predict on!")

    model = load_model()
    assert model is not None

    X_pred_processed = preprocess_predict(X_pred)

    y_pred = model.predict(X_pred_processed)
    prediction_index = np.argmax(y_pred, axis=1)

    return prediction_index[0]


if __name__ == "__main__":
    train()
