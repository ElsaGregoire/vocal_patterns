from fastapi import FastAPI, Query, Request
from numpy import array

app = FastAPI()

# giving random values to the X so i can check it works

X_predict = "Arpegio"
X_trained = ["Arpegio", "Scale", "Other"]


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
async def predict(request: Request):
    data = await request.json()
    float_audio_array_as_list = data["float_audio_array_as_list"]
    float_audio_array = array(float_audio_array_as_list)
    print(float_audio_array.shape)
    return {"prediction": "float_audio_array"}


@app.get("/info")
def get_info():
    return {"ok": True, "message": "API information retrieved successfully"}
