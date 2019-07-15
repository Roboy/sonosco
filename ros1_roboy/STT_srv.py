import logging
from sonosco.ros1.server import SonoscoROS1
from roboy_cognition_msgs.srv import RecognizeSpeech
from roboy_control_msgs.msg import ControlLeds
from webrtcvad_input import VadInput

model_path = "pretrained/deepspeech_final.pth"

CONFIG = {
    'node_name': 'roboy_speech_recognition',
    'processes': 3,
    'subscribers': [
        {
            'name': 'recognition',
            'topic': '/roboy/cognition/speech/recognition',
            'service': RecognizeSpeech,
            'callback': None,

        },
        {
            'name': 'recognition_german',
            'topic': '/roboy/cognition/speech/recognition/german',
            'service': RecognizeSpeech,
            'callback': None,
        }
    ],
    'publishers': [
        {
            'name': 'ledmode',
            'topic': '/roboy/control/matrix/leds/mode',
            'message': ControlLeds,
            'callback': None,
            'kwargs': {
                'queue_size': 3
            }

        },
        {
            'name': 'ledoff',
            'topic': '/roboy/control/matrix/leds/off',
            'message': ControlLeds,
            'callback': None,
            'kwargs': {
                'queue_size': 10
            }
        },
        {
            'name': 'ledfreez',
            'topic': '/roboy/control/matrix/leds/freeze',
            'message': ControlLeds,
            'callback': None,
            'kwargs': {
                'queue_size': 1
            }
        }
    ],

}

def vad_callback():


def main(args=None):
    with SonoscoROS1(CONFIG) as server:
        pass


if __name__ == '__main__':
    logging.basicConfig()
    main()
