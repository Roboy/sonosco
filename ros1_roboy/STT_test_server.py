#!/usr/bin/env python3
import logging

from sonosco.inputs.audio import SonoscoAudioInput
from sonosco.models import DeepSpeech2
from sonosco.models.deepspeech2_inference import DeepSpeech2Inference
from sonosco.ros1.server import SonoscoROS1
from roboy_cognition_msgs.srv import RecognizeSpeech
from roboy_control_msgs.msg import ControlLeds

model_path = "pretrained/deepspeech_final.pth"
audio_path = "test_audio.wav"


class TestAudioInput(SonoscoAudioInput):

    def request_audio(self, *args, **kwargs):
        with open(audio_path, 'rb') as audio_file:
            return audio_file.read()


asr = DeepSpeech2Inference(DeepSpeech2.load_model(model_path))
test_input = TestAudioInput()


def test_callback(request, publishers):
    audio = test_input.request_audio()
    publishers['test'].publish(1, 2)
    return asr.infer(audio)


CONFIG = {
    'node_name': 'roboy_speech_recognition',
    'workers': 5,
    'subscribers': [
        {
            'name': 'recognition',
            'topic': '/roboy/cognition/speech/recognition',
            'service': RecognizeSpeech,
            'callback': test_callback

        }
    ],
    'publishers': [
        {
            'name': 'test',
            'topic': '/roboy/test',
            'message': ControlLeds,
            'kwargs': {
                'queue_size': 3
            }

        }
    ],

}


def main(args=None):
    with SonoscoROS1(CONFIG) as server:
        server.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
