FROM missxa/melodic-dashing-roboy

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install build-essential libssl1.0.0 libasound2 -y
RUN apt-get install python3-pip python3-yaml libsndfile1 portaudio19-dev python3-pyaudio -y
#
RUN pip3 install certifi==2019.3.9 \
                 chardet==3.0.4 \
                 idna==2.8 \
                 Pillow==6.0.0 \
                 PyAudio==0.2.11 \
                 PyYAML==5.1 \
                 requests==2.21.0 \
                 urllib3==1.24.3 \
                 opencv-python \
                 webrtcvad \
                 monotonic \
                 SpeechRecognition \
                 dataclasses \
                 python-Levenshtein \
                 rospkg \
                 catkin_pkg \
                 librosa \
                 ipdb

RUN pip3 install torch
