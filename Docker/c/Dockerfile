FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ Asia/Tokyo

RUN apt-get -y update && apt-get -y upgrade && \
	apt-get install -y --no-install-recommends curl  \
	build-essential \
	zlib1g-dev \
	libffi-dev

RUN mkdir /tmp/src
WORKDIR /tmp/src

COPY src ./