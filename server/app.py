import librosa
import torch
import json
import os
import logging
import tempfile

from flask_cors import CORS
from flask import Flask, render_template, make_response
from flask_socketio import SocketIO, emit
from uuid import uuid1
from utils import get_config, create_pseudo_db, create_session_dir
from model_loader import load_models
from sonosco.common.path_utils import try_create_directory
from external.model_factory import create_external_model
from concurrent.futures import ThreadPoolExecutor

EXTERNAL_MODELS = {"microsoft": None}
REFERENCE_MODEL_ID_KEY = "reference_id"

app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
app.logger.setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
CORS(app)
socketio = SocketIO(app, ping_timeout=120, ping_interval=60)

config = get_config('config.yaml')

device = torch.device("cpu")
loaded_models = load_models(config['models'])

db_path = create_pseudo_db(config['audio_database_path'])
session_dir = create_session_dir(config['sonosco_home'])

logging.basicConfig(level=logging.WARNING, format="ROS RESPONSE: %(message)s")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    """Serve the index HTML"""
    response = make_response(render_template('index.html'))
    response.set_cookie("session_id", str(uuid1()))
    return response


@socketio.on('transcribe')
def on_transcribe(wav_bytes, model_ids):
    temp_audio_file = None

    try:
        # with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        #     temp_audio_file.write(wav_bytes)

        with open("/ros1/to_trans.wav", 'wb') as temp_audio_file:
            temp_audio_file.write(wav_bytes)

        output = dict()
        reference_model_id = model_ids[0]

        with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:

            for model_id in model_ids:

                if model_id in EXTERNAL_MODELS:
                    if EXTERNAL_MODELS[model_id] is None:
                        external_model = create_external_model(model_id, config['sample_rate'])
                        EXTERNAL_MODELS[model_id] = external_model
                    else:
                        external_model = create_external_model(model_id, config['sample_rate'])

                    future = pool.submit(external_model.recognize, temp_audio_file.name)

                else:
                    inference = loaded_models[model_id]
                    future = pool.submit(inference.infer_from_path, temp_audio_file.name)

                output[model_id] = future

            socketio.sleep(0)

        ref_value = None

        for model_id in output.keys():
            output[model_id] = output[model_id].result()
            if model_id == reference_model_id:
                ref_value = output[model_id]

        output[REFERENCE_MODEL_ID_KEY] = ref_value

        emit("transcription", output)

    finally:
        if temp_audio_file:
            pass
            # os.unlink(temp_audio_file.name)


@socketio.on('saveSample')
def on_save_sample(wav_bytes, transcript, user_id):
    if wav_bytes is None:
        return

    path_to_user_data = os.path.join(db_path, "web_collected", user_id)
    try_create_directory(path_to_user_data)
    code = uuid1()
    sample_path = os.path.join(path_to_user_data, str(code))

    try_create_directory(sample_path)
    path_to_wav = os.path.join(sample_path, "audio.wav")
    path_to_txt = os.path.join(sample_path, "transcript.txt")
    temp_audio_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
            temp_audio_file.write(wav_bytes)
        loaded, sr = librosa.load(temp_audio_file.name, sr=config['sample_rate'])
        librosa.output.write_wav(path_to_wav, loaded, config['sample_rate'])
    finally:
        if temp_audio_file:
            os.unlink(temp_audio_file.name)

    with open(path_to_txt, "w") as txt_file:
        txt_file.write(str(transcript))


@app.route('/get_models')
def get_models():
    models = config['models']
    model_dict = {model['id']: model['name'] for model in models}
    return json.dumps(model_dict)


if __name__ == '__main__':
    # socketio.run(app, host='0.0.0.0', certfile='cert.pem', keyfile='key.pem', debug=False)
    socketio.run(app, host='0.0.0.0', debug=False)
