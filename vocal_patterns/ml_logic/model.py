from tensorflow import keras
from keras import Model, optimizers
from keras.callbacks import EarlyStopping
import numpy as np


def init_model(input_shape: tuple) -> Model:
    print("✅ Model initialized")


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
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
    batch_size=256,
    patience=2,
    # validation_data=None,  # overrides validation_split
    validation_split=0.3,
) -> tuple[Model, dict]:
    es = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        X,
        y,
        # validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    return model, history


def evaluate_model(model, data):
    """Evaluates a model"""
    pass
