import os

import logging
import rclpy

from sonosco.ros2.server import SonoscoROS2
from sonosco.models.deepspeech2 import DeepSpeech2
from sonosco.models.deepspeech2_inference import DeepSpeech2Inference


from sonosco.common.utils import setup_logging
from utils import get_config

LOGGER = logging.getLogger("ros2")

model_path = "pretrained/deepspeech_final.pth"


def main(args=None):
    rclpy.init(args=args)

    stt = SonoscoROS2(get_config(), DeepSpeech2Inference(DeepSpeech2.load_model(model_path)))

    while rclpy.ok():
        rclpy.spin_once(stt)

    rclpy.shutdown()


if __name__ == '__main__':
    setup_logging(LOGGER)
    main()
