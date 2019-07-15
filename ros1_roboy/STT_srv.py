import logging
from sonosco.ros1.server import SonoscoROS1
from utils import get_config

model_path = "pretrained/deepspeech_final.pth"


def main(args=None):
    with SonoscoROS1(get_config()) as server:
        pass


if __name__ == '__main__':
    logging.basicConfig()
    main()
