import io

import librosa
import torch
import json
import os
import soundfile as sf

import numpy as np

from flask_cors import CORS

from flask import Flask, render_template, make_response, request
from flask_socketio import SocketIO, emit
from uuid import uuid1
from utils import get_config, transcribe, create_pseudo_db, create_session_dir
from model_loader import load_models
from sonosco.common.path_utils import try_create_directory
from external.model_factory import create_external_model
from concurrent.futures import ThreadPoolExecutor

EXTERNAL_MODELS = {"microsoft": None}

app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
CORS(app)
socketio = SocketIO(app, ping_timeout=120, ping_interval=60)

config = get_config('config.yaml')

device = torch.device("cpu")
loaded_models = load_models(config['models'])

db_path = create_pseudo_db(config['audio_database_path'])
session_dir = create_session_dir(config['sonosco_home'])


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

        socketio.sleep(0)

    for model_id in output.keys():
        output[model_id] = output[model_id].result()

    emit("transcription", output)


@socketio.on('saveSample')
def on_save_sample(wav_bytes, transcript, user_id):
    if wav_bytes is None:
        return

    path_to_user_data = os.path.join(db_path, "web_collected", user_id)
    try_create_directory(path_to_user_data)
    code = uuid1()
    sample_path = os.path.join(path_to_user_data, str(code))

    try_create_directory(sample_path)
    path_to_wav = os.path.join(sample_path, f"audio.webm")
    path_to_txt = os.path.join(sample_path, f"transcript.txt")

    with open(path_to_wav, "wb") as wav_file:
        wav_file.write(wav_bytes)
    # loaded, sr = librosa.load(path_to_wav, sr=16000)
    # print(f"2Loaded: {len(loaded)}, bytes: {len(list(wav_bytes))}")
    # audio_ints = np.frombuffer(wav_bytes[0:len(wav_bytes) - len(wav_bytes) % 32], dtype=np.int32)
    # normalized = audio_ints / np.max(np.abs(audio_ints))
    # data, samplerate=  sf.read(io.BytesIO(wav_bytes), channels=1, samplerate=16000,
    #                        subtype='FLOAT',format='RAW',dtype='float32')
    # librosa.output.write_wav(path_to_wav, loaded, 16000)
    with open(path_to_txt, "w") as txt_file:
        txt_file.write(str(transcript))


@app.route('/get_models')
def get_models():
    models = config['models']
    model_dict = {model['id']: model['name'] for model in models}
    return json.dumps(model_dict)


if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', certfile='cert.pem', keyfile='key.pem', debug=False)

    socketio.run(app, host='0.0.0.0', debug=True)
