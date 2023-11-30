from fastapi import FastAPI, Query, Request
from numpy import array

from vocal_patterns.interface.main import predict
from vocal_patterns.ml_logic.preprocessor import preprocess_predict

app = FastAPI()


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
    float_audio_array = array(float_audio_array_as_list)
    processed_spectrograms = preprocess_predict(float_audio_array)

    raw_predictions = []
    for spectrogram in processed_spectrograms:
        prediction = predict(spectrogram)
        raw_predictions.append(prediction)

    print(raw_predictions)

    # Get the most common prediction
    prediction = max(set(raw_predictions), key=raw_predictions.count)

    return {"prediction": int(prediction)}


@app.get("/info")
def get_info():
    return {"ok": True, "message": "API information retrieved successfully"}
