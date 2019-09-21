FROM ros1_sonosco_base

SHELL ["/bin/bash", "-c"]

WORKDIR /ros1
COPY . .

RUN apt update
RUN apt install ffmpeg iputils-ping nodejs nodejs-dev node-gyp libssl1.0-dev npm -y

RUN git clone https://github.com/Roboy/sonosco.git
RUN cd sonosco; git checkout develop; pip3 install -e .
RUN chmod +x STT_server.py
RUN chmod +x STT_client.py
RUN cd sonosco/server/frontend; npm install; npm run build
#RUN source ~/melodic_ws/devel/setup.bash
#RUN roscore &
#ENTRYPOINT [ "bash", "-c", "source /opt/ros/melodic/setup.bash; python3 STT_srv.py" ]
