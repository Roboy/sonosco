import torch
import json
import os
import time
from flask_cors import CORS

from flask import Flask, render_template, make_response, request
from flask_socketio import SocketIO, emit
from uuid import uuid1
from utils import get_config, transcribe
from model_loader import load_models
from sonosco.common.path_utils import try_create_directory
from external.model_factory import create_external_model
from concurrent.futures import ThreadPoolExecutor

EXTERNAL_MODELS = {"microsoft": None}

app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
CORS(app)
socketio = SocketIO(app)

config = get_config('config.yaml')
tmp_dir = os.path.join(os.path.expanduser("~"), ".sonosco")
session_dir = os.path.join(tmp_dir, "session")
try_create_directory(session_dir)

device = torch.device("cpu")
loaded_models = load_models(config['models'])


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    """Serve the index HTML"""
    response = make_response(render_template('index.html'))
    response.set_cookie("session_id", str(uuid1()))
    return response


@socketio.on('transcribe')
def on_transcribe(wav_bytes, model_ids):
    session_id = request.cookies.get("session_id")
    audio_path = os.path.join(session_dir, f"{session_id}.wav")

    with open(audio_path, "wb") as file:
        file.write(wav_bytes)

    output = dict()

    with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:

        for model_id in model_ids:

            if model_id in EXTERNAL_MODELS:
                if EXTERNAL_MODELS[model_id] is None:
                    external_model = create_external_model(model_id)
                    EXTERNAL_MODELS[model_id] = external_model
                else:
                    external_model = create_external_model(model_id)

                future = pool.submit(external_model.recognize, audio_path)

            else:
                model_config = loaded_models[model_id]
                future = pool.submit(transcribe, model_config, audio_path, device)

            output[model_id] = future

    for model_id in output.keys():
        output[model_id] = output[model_id].result()

    emit("transcription", output)


@socketio.on('saveSample')
def on_save_sample(wav_bytes, transcript, userID):
    path_to_userdata = os.path.join(os.path.expanduser("~"), "data/temp/" + userID)
    try_create_directory(path_to_userdata)
    counter = len([x[0] for x in os.walk(path_to_userdata) if os.path.isdir(x[0])])-1

    path_to_sample = os.path.join(path_to_userdata, str(counter))
    try_create_directory(path_to_sample)

    path_to_wav = os.path.join(path_to_sample, "audio.wav")
    path_to_txt = os.path.join(path_to_sample, "transcript.txt")
    with open(path_to_wav, "wb") as wav_file:
        wav_file.write(wav_bytes)
    with open(path_to_txt, "w") as txt_file:
        txt_file.write(str(transcript))


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