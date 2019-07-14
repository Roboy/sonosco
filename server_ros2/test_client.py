from time import sleep

import rclpy

from sonosco.common.utils import setup_logging


import logging

from roboy_cognition_msgs.msg import RecognizedSpeech
from sonosco.common.constants import SONOSCO

LOGGER = logging.getLogger(SONOSCO)
LOGGER.setLevel(logging.INFO)




def main():
    rclpy.init()
    node = rclpy.create_node('odas_speech_recognition')
    publisher = node.create_publisher(RecognizedSpeech, '/roboy/cognition/speech/recognition')
    while rclpy.ok():
        with open("audio.wav", "rb") as file:
            request = RecognizedSpeech()
            request.text = file
            publisher.publish(request)
        rclpy.spin_once(node)
        sleep(1)

    rclpy.shutdown()


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
