import socket
import threading
import numpy as np
import webrtcvad
import sys
import logging

from collections import deque
from sonosco.inputs.audio import SonoscoAudioInput

logging.basicConfig(level=logging.INFO)


class MicrophoneClient(SonoscoAudioInput):

    def __init__(self, port=10001, host='172.16.100.219', sample_rate=16000, chunk_size=1024):
        # self.format = pyaudio.paInt16

        self.SAMPLE_WIDTH = 2  # pyaudio.get_sample_size(self.format)  # size of each sample
        self.SAMPLE_RATE = sample_rate  # sampling rate in Hertz
        self.CHUNK = chunk_size
        self.CHANNELS = 1
        self.record = False

        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
        self.PADDING_DURATION_MS = 1000
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.CHUNK_BYTES = self.CHUNK_SIZE * 2
        self.NUM_PADDING_CHUNKS = int(self.PADDING_DURATION_MS / self.CHUNK_DURATION_MS)
        self.NUM_WINDOW_CHUNKS = int(240 / self.CHUNK_DURATION_MS)

        self.stream = MicrophoneClient.BytesLoop()
        self.vad = webrtcvad.Vad(2)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        logging.info("Microphone client connected")

        d = threading.Thread(target=self.write_to_streams)
        self.lock = threading.RLock()
        d.setDaemon(True)
        d.start()

    def write_to_streams(self):
        logging.info("Started mic client deamon")
        while True:
            data = self.s.recv(self.CHUNK)
            if self.record:
                data = np.frombuffer(data, dtype=np.int16)
                self.stream.write(data.tobytes())

    def request_audio(self, *args, **kwargs):
        try:
            got_a_sentence = False

            ring_buffer = deque(maxlen=self.NUM_PADDING_CHUNKS)
            triggered = False
            voiced_frames = []
            ring_buffer_flags = [0] * self.NUM_WINDOW_CHUNKS

            ring_buffer_index = 0
            buffer_in = ''

            logging.info("* recording")

            while not got_a_sentence:  # and not leave:
                chunk = self.stream.read(self.CHUNK_SIZE)
                active = self.vad.is_speech(chunk, self.RATE)
                logging.info('1' if active else '0')
                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index += 1
                ring_buffer_index %= self.NUM_WINDOW_CHUNKS
                if not triggered:
                    ring_buffer.append(chunk)
                    num_voiced = sum(ring_buffer_flags)
                    if num_voiced > 0.5 * self.NUM_WINDOW_CHUNKS:
                        logging.info('+')
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(chunk)
                    ring_buffer.append(chunk)
                    num_unvoiced = self.NUM_WINDOW_CHUNKS - sum(ring_buffer_flags)
                    if num_unvoiced > 0.9 * self.NUM_WINDOW_CHUNKS:
                        sys.stdout.write('-')
                        triggered = False
                        got_a_sentence = True

            data = b''.join(voiced_frames)
            logging.info("* done recording")
            return data

        except Exception as e:
            logging.exception(f"Mic client exception {e}")
            raise e

    def __enter__(self):
        # assert self.stream is None, "This audio source is already inside a context manager"
        self.record = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # self.stream = None
        self.record = False

    class BytesLoop:
        def __init__(self, s=b''):
            self.buffer = s
            self.lock = threading.RLock()

        def read(self, n=-1):
            # print("read %i"%n)
            while len(self.buffer) < n:
                pass
            # print("got enough data")
            # self.lock.acquire()

            chunk = self.buffer[:n]
            self.buffer = self.buffer[n:]
            # self.lock.release()
            return chunk

        def write(self, s):
            # print("write %i"%len(s))
            # self.lock.acquire()
            self.buffer += s
            # self.lock.release()
