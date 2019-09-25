import os
import logging
from typing import Callable

import rospy

from concurrent.futures.thread import ThreadPoolExecutor

from inference import SonoscoASR
from inputs import SonoscoAudioInput
from sonosco.inference.dummp_asr import DummyASR
from sonosco.inputs.dummy_input import DummyInput
from sonosco.common.constants import SONOSCO

LOGGER = logging.getLogger(SONOSCO)


class SonoscoROS1:
    def __init__(self, config: dict,
                 default_asr_interface: SonoscoASR = DummyASR(),
                 default_audio_interface: SonoscoAudioInput = DummyInput()):
        """
        Sonosco handler of ROS1 server
        Args:
            config: configuration for the server
            default_asr_interface: default interface to user for ASR
            default_audio_interface: default audio source
        """
        LOGGER.info(f"Sonosco ROS1 server running running with PID: {os.getpid()}")
        self.node_name = config['node_name']

        self.executor = ThreadPoolExecutor(max_workers=config.get('workers', 2))
        self.default_audio_interface = default_audio_interface
        self.default_asr_interface = default_asr_interface

        self.publishers = {entry['name']:
                               rospy.Publisher(entry['topic'],
                                               entry['message'],
                                               **entry.get('kwargs', {}))
                           for entry in config['publishers']}

        self.subscribers = {entry['name']:
                                rospy.Service(entry['topic'],
                                              entry['service'],
                                              self.__callback_async_wrapper(
                                                  entry.get('callback', self.__default_callback)),
                                              **entry.get('kwargs', {}))
                            for entry in config['subscribers']}

        LOGGER.info("Sonosco ROS1 server is ready!")

    def run(self) -> None:
        """
        Starts the server

        """
        rospy.init_node(self.node_name)
        rospy.spin()

    def __default_callback(self, request: any, publishers: any) -> str:
        """
        Default callback for handling ros requests, uses default asr and audio input
        Args:
            request: not used
            publishers: not used

        Returns: transcription from default ASR

        """
        LOGGER.info('Incoming Audio')
        audio = self.default_audio_interface.request_audio()
        return self.default_asr_interface.infer(audio)

    def __callback_async_wrapper(self, callback: Callable) -> Callable:
        """
        Helper callback wrapper
        Args:
            callback: callback to wrap

        Returns: wrapper

        """
        publishers = self.publishers

        def wrapper(request):
            return self.executor.submit(callback, request, publishers).result()

        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.executor.shutdown()
        except Exception as e:
            LOGGER.error(f"Exception while closing thread pool {e}")
