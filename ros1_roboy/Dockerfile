FROM ros1_sonosco_base

SHELL ["/bin/bash", "-c"]

WORKDIR /ros1
COPY . .

RUN apt update
RUN apt install ffmpeg -y

RUN git clone https://github.com/Roboy/sonosco.git
RUN cd sonosco; git checkout SWA-84-ros1-integration; pip3 install .
RUN chmod +x STT_server.py
RUN chmod +x STT_client.py
#RUN . ~/melodic_ws/devel/setup.bash
#RUN roscore &
#ENTRYPOINT [ "bash", "-c", "source /opt/ros/melodic/setup.bash; python3 STT_srv.py" ]
