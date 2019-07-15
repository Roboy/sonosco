import torch
import json
from flask_cors import CORS

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from utils import get_config, transcribe
from model_loader import load_models
from external.model_factory import create_external_model

EXTERNAL_MODELS = {"microsoft": None}

app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
CORS(app)
socketio = SocketIO(app)

config = get_config('../server_transcriptor/config.yaml')

audio_path = "audio.wav"

device = torch.device("cpu")
loaded_models = load_models(config['models'])


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    """Serve the index HTML"""
    return render_template('index.html')


@socketio.on('transcribe')
def on_transcribe(wav_bytes, model_ids):
    with open(audio_path, "wb") as file:
        file.write(wav_bytes)

    output = dict()

    for model_id in model_ids:

        if model_id in EXTERNAL_MODELS:
            if EXTERNAL_MODELS[model_id] is None:
                external_model = create_external_model(model_id)
                EXTERNAL_MODELS[model_id] = external_model
            else:
                external_model = create_external_model(model_id)

            transcription = external_model.recognize(audio_path)

        else:
            model_config = loaded_models[model_id]
            transcription = transcribe(model_config, audio_path, device)

        output[model_id] = transcription

    emit("transcription", output)


@app.route('/get_models')
def get_models():
    models = config['models']
    model_dict = {model['id']: model['name'] for model in models}
    return json.dumps(model_dict)


if __name__ == '__main__':
    # eventlet.wsgi.server(eventlet.wrap_ssl(eventlet.listen(("0.0.0.0", 5000)), certfile='cert.pem',
    #                                        keyfile='key.pem', server_side=True), app)
    # socketio.run(app, host='0.0.0.0', certfile='cert.pem', keyfile='key.pem', debug=True)
    socketio.run(app, host='0.0.0.0', debug=True)
