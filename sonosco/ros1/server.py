import multiprocessing
import os

import logging
import rospy

from sonosco.inference.dummp_asr import DummyASR
from sonosco.inputs.dummy_input import DummyInput
from sonosco.common.constants import SONOSCO

LOGGER = logging.getLogger(SONOSCO)


# TODO: Add possibility to specify asr interface per topi
# TODO: Create new service type with audio input, or handle the audio input (mic, odas) here (decouple with interfaces)
class SonoscoROS1:
    def __init__(self, config, default_asr_interface=DummyASR(), default_audio_interface=DummyInput()):
        LOGGER.info(f"Sonosco ROS2 server running running with PID: {os.getpid()}")
        self.pool = multiprocessing.Pool(processes=config.get('processes', 2))
        self.default_audio_interface = default_audio_interface
        self.default_asr_interface = default_asr_interface
        self.subscribers = {entry['name']:
                                rospy.Service(entry['topic'],
                                              entry['service'],
                                              self.__callback_async_wrapper(
                                                  entry.get('callback', self.__default_callback)),
                                              **entry.get(['kwargs', {}]))
                            for entry in config['subscribers']}

        self.publishers = {entry['name']:
                               rospy.Publisher(entry['topic'],
                                               entry['message'],
                                               **entry.get(['kwargs', {}]))
                           for entry in config['publishers']}
        self.node_name = config['node_name']
        LOGGER.info("Sonosco ROS2 server is ready!")

    def run(self):
        rospy.init_node(self.node_name)
        rospy.spin()

    def __default_callback(self, request, publishers):
        LOGGER.info('Incoming Audio')
        audio = self.default_audio_interface.request_audio()
        return self.default_asr_interface.infer(audio)

    def __callback_async_wrapper(self, callback):
        def wrapper(request):
            return self.pool.apply(callback, (request, self.publishers))

        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.pool.close()
        except Exception as e:
            LOGGER.error(f"Exception while closing process pool {e}")
            self.pool.terminate()
        rospy.core.signal_shutdown("Closing ROS1 Sonosco server")
