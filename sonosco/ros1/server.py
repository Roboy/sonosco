import multiprocessing
import os

import logging
import rospy

from ros1.callback import CallbackWrapper
from sonosco.inference.dummp_asr import DummyASR
from sonosco.inputs.dummy_input import DummyInput
from sonosco.common.constants import SONOSCO

LOGGER = logging.getLogger(SONOSCO)
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager


class SonoscoROS1:
    def __init__(self, config, default_asr_interface=DummyASR(), default_audio_interface=DummyInput()):
        LOGGER.info(f"Sonosco ROS2 server running running with PID: {os.getpid()}")

        self.config = config
        self.node_name = config['node_name']
        self.pool = multiprocessing.Pool(initializer=lambda: rospy.init_node(self.node_name),
                                         processes=config.get('processes', 2))

        self.default_audio_interface = default_audio_interface
        self.default_asr_interface = default_asr_interface
        #
        # BaseManager.register('Publisher', rospy.Publisher)
        # manager = BaseManager()
        # manager.start()

        self.publishers = {entry['name']:
                               rospy.Publisher(entry['topic'],
                                                 entry['message'],
                                                 **entry.get('kwargs', {}))
                           for entry in config['publishers']}

        self.subscribers = {entry['name']:
                                rospy.Service(entry['topic'],
                                              entry['service'],
                                              self.__callback_async_wrapper(
                                                  self.__callback_factory(entry.get('callback', self.__default_callback))),
                                              **entry.get('kwargs', {}))
                            for entry in config['subscribers']}


        LOGGER.info("Sonosco ROS2 server is ready!")

    def run(self):
        rospy.init_node(self.node_name)
        rospy.spin()


    def __callback_factory(self, callback):
        callback_wrapper = CallbackWrapper(self.config['publishers'])
        callback_wrapper.register_callback(callback)
        return callback_wrapper.service_callback


    def __default_callback(self, request, publishers):
        LOGGER.info('Incoming Audio')
        audio = self.default_audio_interface.request_audio()
        return self.default_asr_interface.infer(audio)

    def __callback_async_wrapper(self, callback):
        publishers = self.publishers

        def wrapper(request):
            return self.pool.apply_async(callback, (request, publishers))

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
