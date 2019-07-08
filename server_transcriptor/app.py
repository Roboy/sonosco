import logging
import os
import torch
# import eventlet
import json
from flask_cors import CORS

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor
from utils import get_config
from model_loader import load_models

app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
CORS(app)
socketio = SocketIO(app)

config = get_config('../server_transcriptor/config.yaml')

# model_path = "../pretrained/librispeech_pretrained.pth"
audio_path = "audio.wav"

device = torch.device("cpu")
# model = DeepSpeech2.load_model(model_path)
loaded_models = load_models(config['models'])


# decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
# processor = AudioDataProcessor(**model.audio_conf, normalize=True)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    """Serve the index HTML"""
    return render_template('index.html')


@socketio.on('record')
def on_create(wav_bytes, model_ids):
    with open("audio.wav", "wb") as file:
        file.write(wav_bytes)
        output = {}
    for model_id in model_ids:
        # TODO: move to load script
        model_config = loaded_models[model_id]
        spect = model_config['processor'].parse_audio(audio_path)
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        spect = spect.to(device)
        input_sizes = torch.IntTensor([spect.size(3)]).int()
        out, output_sizes = model_config['model'](spect, input_sizes)
        decoded_output, decoded_offsets = model_config['decoder'].decode(out, output_sizes)
        output[model_id] = decoded_output[0]
    emit("transcription", output)


@app.route('/get_models')
def get_models():
    models = config['models']
    # model_list = [{key: val for key, val in entry.items() if key in ['id', 'name']} for entry in models]
    model_dict = {model['id']: model['name'] for model in models}
    # return json.dumps(model_list)
    return json.dumps(model_dict)


if __name__ == '__main__':
    # eventlet.wsgi.server(eventlet.wrap_ssl(eventlet.listen(("0.0.0.0", 5000)), certfile='cert.pem',
    #                                        keyfile='key.pem', server_side=True), app)
    # socketio.run(app, host='0.0.0.0', certfile='cert.pem', keyfile='key.pem', debug=True)
    socketio.run(app, host='0.0.0.0', debug=True)
