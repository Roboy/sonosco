import subprocess
import numpy as np
import librosa


def get_duration(file_path: str) -> float:
    """
    gets duration of audio using sox
    Args:
        file_path:

    Returns: duration

    """
    return float(subprocess.check_output([f'sox -D "{file_path.strip()}"'], shell=True))


def transcode_recording(source: str, destination: str, sample_rate: int) -> None:
    subprocess.call([f"sox {source} -r {sample_rate} -b 16 -c 1 {destination}"], shell=True)


def transcode_recordings_an4(raw_path: str, wav_path: str, sample_rate: str) -> None:
    """
    transcode recordings using sox
    Args:
        raw_path:
        wav_path:
        sample_rate:

    """
    subprocess.call([f'sox -t raw -r {sample_rate} -b 16 -e signed-integer -B -c 1 \"{raw_path}\" \"{wav_path}\"'],
                    shell=True)


def transcode_recordings_ted3(source: str, destination: str, start_time: int, end_time: int, sample_rate: int) -> None:
    """
    transcode recordings using sox
    Args:
        source:
        destination:
        start_time:
        end_time:
        sample_rate:

    Returns:

    """
    subprocess.call([f"sox {source}  -r {sample_rate} -b 16 -c 1 {destination} trim {start_time} ={end_time}"],
                    shell=True)


def shift(audio: np.ndarray, n_samples: int = 1600) -> np.ndarray:
    """
    shift audio by n_samples
    Args:
        audio:
        n_samples:

    Returns: shifted audio

    """
    return np.roll(audio, n_samples)


def stretch(audio: np.ndarray, rate: int = 1) -> np.ndarray:
    """
    stretches the audio by rate
    Args:
        audio:
        rate:

    Returns: stretched audio

    """
    stretched_audio = librosa.effects.time_stretch(audio, rate)
    return stretched_audio


def pitch_shift(audio: np.ndarray, sample_rate: int = 16000, n_steps: float = 3.0):
    """
    shifts pitch of the audio
    Args:
        audio:
        sample_rate:
        n_steps:

    Returns: shifted audio

    """
    stretched_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    return stretched_audio
