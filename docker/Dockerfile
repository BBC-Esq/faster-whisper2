FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
WORKDIR /root
RUN apt-get update -y && apt-get install -y python3-pip
COPY infer.py jfk.flac ./
RUN pip3 install faster-whisper2
CMD ["python3", "infer.py"]
