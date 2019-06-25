import os

from roboy_cognition_msgs.msg import RecognizedSpeech
from roboy_cognition_msgs.srv import RecognizeSpeech

from asr_interface import IAsr
import rclpy
from rclpy.node import Node


class SonoscoROS2(Node):
    def __init__(self):
        super().__init__('stt')
        self.publisher = self.create_publisher(RecognizedSpeech, '/roboy/cognition/speech/recognition')
        self.srv = self.create_service(RecognizeSpeech, '/roboy/cognition/speech/recognition/recognize', self.asr_callback)
        print("Ready to /roboy/cognition/speech/recognition/recognize")
        print(f"Roboy Sonosco running with PID: {os.getpid()}")
        self.i=IAsr()
        print(f"Status: Speech recognition is ready now!")
        print("Roboy Sonosco is ready!")

    def asr_callback(self, request, response):
        response.success = True
        self.get_logger().info('Incoming Audio')
        msg = RecognizedSpeech()
        self.i.inference_audio(request)
        self.publisher.publish(msg)
        return response


def main(args=None):
    rclpy.init(args=args)

    stt = SonoscoROS2()

    while rclpy.ok():
        rclpy.spin_once(stt)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
