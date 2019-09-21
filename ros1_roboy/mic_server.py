import pyaudio
import socket
import select
import click
import logging


@click.command()
@click.option('--host', '-h', default="0.0.0.0", help='Host')
@click.option('--port', '-p', default=10002, help='Port', type=int)
def main(host, port):
    """
    Creates websocket server that publishes audio from mic in chunks on gives host:port address
    Args:
        host: ip of server
        port: port of server

    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK_DURATION_MS = 10  # supports 10, 20 and 30 (ms)
    CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)

    audio = pyaudio.PyAudio()

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host, port)
    serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversocket.bind(server_address)
    serversocket.listen(5)

    def callback(in_data, frame_count, time_info, status):
        for s in read_list[1:]:
            s.send(in_data)

        return (None, pyaudio.paContinue)

    # start Recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=callback)
    # stream.start_stream()

    read_list = [serversocket]
    logging.info(f"Recording on {server_address}")

    try:
        while True:
            readable, writable, errored = select.select(read_list, [], [])
            for s in readable:
                if s is serversocket:
                    (clientsocket, address) = serversocket.accept()
                    read_list.append(clientsocket)
                    logging.info(f"Connection from {address}")
                else:
                    data = None
                    try:
                        data = s.recv(1024)
                    except:
                        if s in read_list:
                            read_list.remove(s)
                    if not data:
                        if s in read_list:
                            read_list.remove(s)
    except KeyboardInterrupt:
        pass
    finally:
        serversocket.close()
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
