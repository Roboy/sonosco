FROM ros2_sonosco_base

SHELL ["/bin/bash", "-c"]

WORKDIR /ros2
COPY . .

RUN git clone https://github.com/Roboy/sonosco.git
RUN cd sonosco; git checkout SWA-74-ros2-integration; pip3 install .
#RUN source /opt/ros/melodic/setup.bash
#ENTRYPOINT [ "bash", "-c", "source /opt/ros/melodic/setup.bash; python3 STT_srv.py" ]
