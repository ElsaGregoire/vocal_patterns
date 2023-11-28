from fastapi import FastAPI, Query


app = FastAPI()

# Define a root `/` endpoint

@app.get('/')
def root():
    params = {
    'greeting': '''
    Hello,
    Welcome to the Vocal Pattern App'''
}
    return params.get('greeting')

@app.get('/predict_sound')
def predict(type_):
    if X_predict == X_trained[0]:
        return {'You are singing': type_ == 'an Arpegio'}
    if X_predict == X_trained[1]:
        return {'You are singing': type_ == 'a Scale'}
    if X_predict == X_trained[2]:
        return {'You are singing': type_ == 'other type of sound (melody, long notes, improvisation...)'}

@app.get('/predict_file')
def predict(type_):
    if X_predict == X_trained[0]:
        return {'You are singing': type_ == 'an Arpegio'}
    if X_predict == X_trained[1]:
        return {'You are singing': type_ == 'a Scale'}
    if X_predict == X_trained[2]:
        return {'You are singing': type_ == 'other type of sound (melody, long notes, improvisation...)'}


@app.get('/info')
def get_info():
    return {'ok': True, 'message': 'API information retrieved successfully'}
