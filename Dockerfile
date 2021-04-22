FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime


RUN apt update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin" apt install -y libopenblas-base libomp-dev git-all

RUN mkdir -p /biencoder
WORKDIR /biencoder

COPY src/biencodernel ./src/biencodernel/
COPY src/*.py ./src/
COPY requirements.txt .

RUN pip install -r requirements.txt