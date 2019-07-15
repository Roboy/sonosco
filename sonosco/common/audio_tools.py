import subprocess
import numpy as np
import librosa

from .noise_makers import NoiseMaker, GaussianNoiseMaker


def get_duration(file_path):
    return float(subprocess.check_output([f'soxi -D "{file_path.strip()}"'], shell=True))


def transcode_recording(source, destination, sample_rate):
    subprocess.call([f"sox {source} -r {sample_rate} -b 16 -c 1 {destination}"], shell=True)


def transcode_recordings_an4(raw_path, wav_path, sample_rate):
    subprocess.call([f'sox -t raw -r {sample_rate} -b 16 -e signed-integer -B -c 1 \"{raw_path}\" \"{wav_path}\"'],
                    shell=True)


def transcode_recordings_ted3(source, destination, start_time, end_time, sample_rate):
    subprocess.call([f"sox {source}  -r {sample_rate} -b 16 -c 1 {destination} trim {start_time} ={end_time}"],
                    shell=True)


def shift(audio, n_samples=1600):
    return np.roll(audio, n_samples)


def stretch(audio, rate=1):
    stretched_audio = librosa.effects.time_stretch(audio, rate)
    return stretched_audio


def pitch_shift(audio, sample_rate=16000, n_steps=3.0):
    stretched_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    return stretched_audio
