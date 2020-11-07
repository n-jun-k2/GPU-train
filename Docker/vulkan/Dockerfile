FROM nvidia/vulkan:1.1.121-cuda-10.1-beta.1-ubuntu18.04

ENV CMAKE_VERSION=3.15.2
ENV CMAKE_MAJOR_VERSION=3.15

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false  -y update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends curl  \
        ca-certificates \
        build-essential \
		zlib1g-dev \
		libffi-dev \
		zip \ 
		unzip \
        wget \
    && apt-get clean \
    && rm -r /var/lib/apt/lists/*

WORKDIR /tmp/src

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    && tar -xvf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    && cp cmake-${CMAKE_VERSION}-Linux-x86_64/bin/cmake /usr/local/bin \
    && mkdir -p /usr/local/share/cmake-${CMAKE_MAJOR_VERSION} \
    && cp -r cmake-${CMAKE_VERSION}-Linux-x86_64/share/cmake-${CMAKE_MAJOR_VERSION}/* /usr/local/share/cmake-${CMAKE_MAJOR_VERSION}

WORKDIR /tmp
RUN rm -rf /tmp/src