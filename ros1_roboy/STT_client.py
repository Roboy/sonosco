#!/usr/bin/env python
import logging
import rospy
import os.path, sys

from roboy_cognition_msgs.srv import RecognizeSpeech

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def stt_client():
    """
    Simple ROS1 client that requests audio transcription
    Returns: Transcribed audio

    """
    rospy.wait_for_service("/roboy/cognition/speech/recognition")
    try:
        stt = rospy.ServiceProxy("/roboy/cognition/speech/recognition", RecognizeSpeech)
        resp = stt()
        logging.info(f"response from stt: {resp.text}")
        return resp.text
    except rospy.ServiceException as e:
        logging.error(f"Service call failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stt_client()
