from keras import optimizers, layers, models
from tensorflow.keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from colorama import Fore, Style
import numpy as np
from typing import Tuple


def init_model(input_shape: Tuple) -> Model:
    """
    Initialize the CNN model
    """
    model = Sequential()
    model.add(
        layers.Conv2D(
            8,
            (5, 5),
            input_shape=input_shape,
            strides=(2, 2),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", activation="relu")
    )
    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")
    )
    model.add(layers.BatchNormalization())

    model.add(
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="relu")
    )
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.001) -> Model:
    """
    Compile the CNN
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("✅ Model compiled")
    return model


def fit_model(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    batch_size=32,
    patience=2,
    validation_split=0.2,
) -> Tuple[Model, dict]:
    es = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_split=validation_split,
        epochs=20,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )
    return model, history


def evaluate_model(
    model: Model, X: np.ndarray, y: np.ndarray, batch_size=32
) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True,
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, accuracy: {round(accuracy, 2)}")

    return metrics
