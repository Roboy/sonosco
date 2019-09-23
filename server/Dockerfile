FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

WORKDIR /work


RUN apt update
RUN apt install ffmpeg \
                iputils-ping \
                nodejs nodejs-dev \
                node-gyp libssl1.0-dev \
                npm \
                build-essential \
                libssl1.0.0 \
                libasound2 \
                python3-pip \
                python3-yaml \
                libsndfile1 \
                portaudio19-dev \
                python3-pyaudio \
                git \
                vim -y

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

RUN git clone https://github.com/Roboy/sonosco.git
RUN cd sonosco; pip3 install -e .
COPY . .
RUN cd frontend; npm install; npm run build


ENTRYPOINT [ "python3",  "app.py" ]


