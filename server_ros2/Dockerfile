FROM missxa/melodic-crystal-roboy

WORKDIR /ros2

RUN apt-get install build-essential libssl1.0.0 libasound2

RUN pip3 install certifi==2019.3.9 \
                 chardet==3.0.4 \
                 idna==2.8 \
                 Pillow==6.0.0 \
                 PyAudio==0.2.11 \
                 PyYAML==5.1 \
                 requests==2.21.0 \
                 urllib3==1.24.3 \
                 opencv-python \
                 click

# install roboy-intelligence-team
RUN pip3 install -r requirements.txt \
  & pip3 install -e .

ENTRYPOINT [ "scripts/entrypoint.sh" ]
