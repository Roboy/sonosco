FROM missxa/melodic-crystal-roboy

WORKDIR /ros2
COPY . .
RUN apt-get install build-essential libssl1.0.0 libasound2 -y
RUN apt-get install portaudio19-dev python-pyaudio python3-pyaudio libsndfile1 -y
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
                 torch \
                 dataclasses \
                 python-Levenshtein \
                 librosa

RUN git clone https://github.com/Roboy/sonosco.git
RUN cd sonosco; git checkout SWA-74-ros2-integration; pip3 install .
ENTRYPOINT [ "python3", "test_client.py" ]
