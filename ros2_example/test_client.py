from time import sleep

import rclpy

import logging

from roboy_cognition_msgs.srv import RecognizeSpeech


# TODO: Create new service type with audio input, or change to empty request flow
def main():
    rclpy.init()
    node = rclpy.create_node('odas_speech_recognition')
    publisher = node.create_publisher(RecognizeSpeech, '/roboy/cognition/speech/recognition')
    while rclpy.ok():
        with open("audio.wav", "rb") as file:
            # TODO: This won't work with current srv layout
            request = RecognizeSpeech()
            request.text = file
            publisher.publish(request)
        rclpy.spin_once(node)
        sleep(1)

    rclpy.shutdown()


if __name__ == '__main__':
    logging.basicConfig()
    main()
