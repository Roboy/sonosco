import logging
import os
import torch

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from sonosco.models import DeepSpeech2
from sonosco.decoders import GreedyDecoder
from sonosco.datasets.processor import AudioDataProcessor


app = Flask(__name__, static_folder="./dist/static", template_folder="./dist")
socketio = SocketIO(app)

model_path = "../pretrained/librispeech_pretrained.pth"
audio_path = "audio.wav"

device = torch.device("cpu")
model = DeepSpeech2.load_model(model_path)
decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
processor = AudioDataProcessor(**model.audio_conf)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def index(path):
    """Serve the index HTML"""
    return render_template('index.html')


@socketio.on('record')
def on_create(wav_bytes):
    with open("audio.wav", "wb") as file:
        file.write(wav_bytes)
    spect = processor.parse_audio(audio_path)
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
    emit("transcription", decoded_output[0])


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True)
