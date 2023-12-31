from fastapi import FastAPI, Query, Request
import numpy as np

from vocal_patterns.interface.main import predict
from vocal_patterns.ml_logic.preprocessor import preprocess_predict
from vocal_patterns.ml_logic.registry import load_model
from vocal_patterns.params import DEPLOYMENT

stage = "Production" if DEPLOYMENT == "local" else "Production"

app = FastAPI()
app.state.model = load_model(stage=stage)


# Define a root `/` endpoint
@app.get("/")
def root():
    params = {
        "greeting": """
    Hello,
    Welcome to Voxalyze"""
    }
    return params.get("greeting")


# Define a new endpoint `/predict` that accepts a sound file and returns the predicted class
@app.post("/predict")
async def pred(request: Request):
    data = await request.json()
    float_audio_array_as_list = data["float_audio_array_as_list"]
    float_audio_array = np.array(float_audio_array_as_list)
    processed_spectrograms = preprocess_predict(
        float_audio_array, model=app.state.model
    )
    raw_predictions = []
    for spectrogram in processed_spectrograms:
        spectrogram_expanded = np.expand_dims(spectrogram, axis=0)
        prediction = predict(
            X_pred_processed=spectrogram_expanded, model=app.state.model
        )
        raw_predictions.append(prediction)
    prediction_map = {
        0: "Arpeggio",
        1: "Other",
        2: "Scale",
    }

    mean_prediction = np.mean(raw_predictions, axis=0)
    prediction = np.argmax(mean_prediction)
    confidence = np.max(mean_prediction) * 100
    prediction_str = prediction_map[prediction]

    ## SHOW MODEL AGUMENTATIONS

    augmentations = app.state.model.augmentations

    return {
        "response": {
            "prediction": str(prediction_str),
            "confidence": int(confidence),
            "timestamp": app.state.model.timestamp,
            "augmentations": augmentations,
        }
    }


@app.get("/info")
def get_info():
    return {"ok": True, "message": "API information retrieved successfully"}
