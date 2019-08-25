from sonosco.inputs.audio import SonoscoAudioInput
import webrtcvad
import collections
import os
import signal
import pyaudio
import sys
import logging

class VadInput(SonoscoAudioInput):

    def __init__(self):
        super().__init__()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
        self.PADDING_DURATION_MS = 1000
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.CHUNK_BYTES = self.CHUNK_SIZE * 2
        self.NUM_PADDING_CHUNKS = int(self.PADDING_DURATION_MS / self.CHUNK_DURATION_MS)
        self.NUM_WINDOW_CHUNKS = int(240 / self.CHUNK_DURATION_MS)
        self.vad = webrtcvad.Vad(2)

        pa = pyaudio.PyAudio()
        self.stream = pa.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              start=False,
                              # input_device_index=2,
                              frames_per_buffer=self.CHUNK_SIZE)
        logging.basicConfig()
        self.logger = logging.getLogger("VadInput")

    def request_audio(self, *args, **kwargs):
        got_a_sentence = False

        ring_buffer = collections.deque(maxlen=self.NUM_PADDING_CHUNKS)
        triggered = False
        voiced_frames = []
        ring_buffer_flags = [0] * self.NUM_WINDOW_CHUNKS

        ring_buffer_index = 0
        buffer_in = ''

        print("* recording")
        self.stream.start_stream()
        while not got_a_sentence:  # and not leave:
            chunk = self.stream.read(self.CHUNK_SIZE)
            active = self.vad.is_speech(chunk, self.RATE)
            sys.stdout.write('1' if active else '0')
            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index += 1
            ring_buffer_index %= self.NUM_WINDOW_CHUNKS
            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.5 * self.NUM_WINDOW_CHUNKS:
                    sys.stdout.write('+')
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

            sys.stdout.flush()

        sys.stdout.write('\n')
        data = b''.join(voiced_frames)

        self.stream.stop_stream()
        print("* done recording")
        return data


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        except Exception as e:
            self.logger.error(f"Exception while closing process pool {e}")
