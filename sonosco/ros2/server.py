import os

import logging

from roboy_cognition_msgs.msg import RecognizedSpeech
from roboy_cognition_msgs.srv import RecognizeSpeech

from rclpy.node import Node

LOGGER = logging.getLogger(SONOSCO)


# TODO: Add possibility to specify asr interface per topic
class SonoscoROS2(Node):
    def __init__(self, config, asr_interface):
        super().__init__('stt')
        LOGGER.info(f"Sonosco ROS2 server running running with PID: {os.getpid()}")

        self.asr_interface = asr_interface
        self.topics = config['topics']
        self.publishers = set(self.create_publisher(RecognizedSpeech, name) for name in self.topics)
        self.subscribers = set(self.create_service(RecognizeSpeech, name, self.asr_callback)
                               for name in self.topics)
        LOGGER.info(f"Topics {self.topics}")
        LOGGER.info("Sonosco ROS2 server is ready!")

    def asr_callback(self, request, response):
        LOGGER.info('Incoming Audio')
        transcription = self.asr_interface.infer(request.text)
        response.text = transcription
        response.success = True
        return response
