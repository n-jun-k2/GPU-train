FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# chmod 1777 /tmpはapt-get updateをうまく行う為
RUN chmod 1777 /tmp \ 
	&& apt-get -y update \
	&& apt-get -y upgrade \
	&& apt-get install -y --no-install-recommends curl  \
		build-essential \
		zlib1g-dev \
		libffi-dev \
		libbz2-dev \
		ca-certificates \
		libsqlite3-dev \
		libssl-dev \
	&& apt-get clean \
    && rm -r /var/lib/apt/lists/*

WORKDIR /src
ENV PYTHON_VERSION 3.11.5
ENV PYTHON_MAJOR_VERSION 3.11

RUN sed -i 's/CipherString = DEFAULT:@SECLEVEL=2/CipherString = DEFAULT@SECLEVEL=0/' /usr/lib/ssl/openssl.cnf \
	&& sed '$a\MinProtocol = None' /usr/lib/ssl/openssl.cnf

RUN curl -OL https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-${PYTHON_VERSION}.tar.xz \
	&& tar xJf Python-${PYTHON_VERSION}.tar.xz \
	&& cd ./Python-${PYTHON_VERSION} \
	&& ./configure --enable-optimizations \
	&& make -s \
	&& make install \
	&& cd /usr/local/bin \
	&& ln -s pip${PYTHON_MAJOR_VERSION} pip \
	&& ln -s python${PYTHON_MAJOR_VERSION}-config python-config\
	&& ln -s idle${PYTHON_MAJOR_VERSION} idle \
	&& ln -s pydoc${PYTHON_MAJOR_VERSION} pydoc \
	&& ln -s python${PYTHON_MAJOR_VERSION} python

# pakage install
COPY requirements.in ./
RUN pip install pip-tools \
	&& pip-compile requirements.in \
	&& pip-sync \
	&& pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# インストール作業で出た一時的なファイルを削除
# WORKDIR /src
# RUN rm -rf ./libsrc/ && rm -rf /src/tmp
