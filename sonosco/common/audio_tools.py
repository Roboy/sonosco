import subprocess


def get_duration(file_path):
    return float(subprocess.check_output([f'soxi -D "{file_path.strip()}"'], shell=True))


def transcode_recording(source, destination, sample_rate):
    subprocess.call([f"sox {source}  -r {sample_rate} -b 16 -c 1 {destination}"], shell=True)
