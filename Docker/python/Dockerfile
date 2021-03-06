FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ Asia/Tokyo

RUN apt-get -y update && apt-get -y upgrade && \
	apt-get install -y --no-install-recommends curl  \
		build-essential \
		zlib1g-dev \
		libffi-dev \
		libbz2-dev \
		ca-certificates \
	&& apt-get clean \
    && rm -r /var/lib/apt/lists/*

WORKDIR /tmp/src

ENV PYTHON_VERSION 3.8.5
ENV PYTHON_MAJOR_VERSION 3.8

# openssl install
RUN curl -OL https://www.openssl.org/source/openssl-1.1.1g.tar.gz \
	&& tar -zxf openssl-1.1.1g.tar.gz \
	&& cd openssl-1.1.1g \
	&& ./config \
	&& make install

# python install
RUN curl -OL https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-${PYTHON_VERSION}.tar.xz \
	&& tar xJf Python-${PYTHON_VERSION}.tar.xz \
	&& cd ./Python-${PYTHON_VERSION} \
	&& ./configure \
	&& make -s \
	&& make install\
	&& pip3.8 install --upgrade pip setuptools \
	&& cd /usr/local/bin \
	&& ln -s idle${PYTHON_MAJOR_VERSION} idle \
	&& ln -s pydoc${PYTHON_MAJOR_VERSION} pydoc \
	&& ln -s python${PYTHON_MAJOR_VERSION} python \
	&& ln -s python${PYTHON_MAJOR_VERSION}-config python-config

# pakage install
COPY requirements.txt ./
RUN pip install -r requirements.txt && \
	pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


# インストール作業で出た一時的なファイルを削除
WORKDIR /tmp
RUN rm -rf ./libsrc/
# Create working directory
WORKDIR /tmp/python-app

COPY packages ./