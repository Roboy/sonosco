#!/usr/bin/env python3
import logging
import signal

from sonosco.models import DeepSpeech2

from sonosco.models.deepspeech2_inference import DeepSpeech2Inference
from sonosco.ros1.server import SonoscoROS1
from roboy_cognition_msgs.srv import RecognizeSpeech
from roboy_control_msgs.msg import ControlLeds
from webrtcvad_input import VadInput
from std_msgs.msg import Empty
from mic_client import MicrophoneClient

model_path = "pretrained/deepspeech_final.pth"

asr = DeepSpeech2Inference(DeepSpeech2.load_model(model_path))
leave = False
got_a_sentence = False


def handle_int(sig, chunk):
    global leave, got_a_sentence

    leave = True
    got_a_sentence = True


signal.signal(signal.SIGINT, handle_int)


def vad_callback(request, publishers):
    msg = ControlLeds()
    msg.mode = 2
    msg.duration = 0
    publishers['ledmode'].publish(msg)
    transcription = ""
    with MicrophoneClient() as audio_input:
        # while not leave:
            audio = audio_input.request_audio()
            transcription = asr.infer(audio)
            msg = Empty()
            # publishers['ledfreez'].publish(msg)

    return transcription


CONFIG = {
    'node_name': 'roboy_speech_recognition',
    'workers': 5,
    'subscribers': [
        {
            'name': 'recognition',
            'topic': '/roboy/cognition/speech/recognition',
            'service': RecognizeSpeech,
            'callback': vad_callback,

        },
        {
            'name': 'recognition_german',
            'topic': '/roboy/cognition/speech/recognition/german',
            'service': RecognizeSpeech,
            'callback': vad_callback,
        }
    ],
    'publishers': [
        {
            'name': 'ledmode',
            'topic': '/roboy/control/matrix/leds/mode',
            'message': ControlLeds,
            'kwargs': {
                'queue_size': 3
            }

        },
        {
            'name': 'ledoff',
            'topic': '/roboy/control/matrix/leds/off',
            'message': ControlLeds,
            'kwargs': {
                'queue_size': 10
            }
        },
        {
            'name': 'ledfreez',
            'topic': '/roboy/control/matrix/leds/freeze',
            'message': ControlLeds,
            'kwargs': {
                'queue_size': 1
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
