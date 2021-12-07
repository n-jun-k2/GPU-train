FROM nvcr.io/nvidia/nvhpc:21.9-devel-cuda_multi-ubuntu20.04

ENV CMAKE_VERSION 3.22.0

WORKDIR /tmp
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends libssl-dev \ 
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
    && tar zxvf cmake-${CMAKE_VERSION}.tar.gz \
    && cd cmake-${CMAKE_VERSION}/ \
    && ./bootstrap \
    && make && make install \
    && echo 'export PATH=$HOME/cmake-${CMAKE_VERSION}/bin/:$PATH' >> ~/.bashrc \
    && . ~/.bashrc

WORKDIR /tmp/projects