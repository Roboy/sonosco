import subprocess


def get_duration(file_path):
    return float(subprocess.check_output([f'soxi -D "{file_path.strip()}"'], shell=True))


def transcode_recording(source, destination, sample_rate):
    subprocess.call([f"sox {source}  -r {sample_rate} -b 16 -c 1 {destination}"], shell=True)

def transcode_recordings_an4(raw_path, wav_path, sample_rate):
    subprocess.call([f'sox -t raw -r {sample_rate} -b 16 -e signed-integer -B -c 1 \"{raw_path}\" \"{wav_path}\"'], shell=True)

def transcode_recordings_ted3(source, destination, start_time, end_time, sample_rate):
    subprocess.call([f"sox {source}  -r {sample_rate} -b 16 -c 1 {destination} trim {start_time} ={end_time}"],shell=True)