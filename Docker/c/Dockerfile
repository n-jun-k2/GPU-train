FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV TZ Asia/Tokyo

ENV BOOST_VERSION 1.70.0
ENV BOOST_FILE_VERSION 1_70_0
ENV GOOGLE_TEST_VERSION 1.8.0
ENV PYTORCH_CUDA_VERSION 100
ENV PYTORCH_VERSION 1.2.0

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false  -y update \
	&& apt-get -y upgrade \
	&& apt-get install -y --no-install-recommends curl  \
		ca-certificates \
		build-essential \
		zlib1g-dev \
		libffi-dev \
		zip \ 
		unzip \
		cmake \
    && apt-get clean \
    && rm -r /var/lib/apt/lists/*

# BoostとGoogle testとPyTorchをダウンロードし解凍する。
WORKDIR /tmp/libsrc
RUN curl -OL https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_FILE_VERSION}.tar.gz \
    && tar -zxvf boost_${BOOST_FILE_VERSION}.tar.gz \
    && curl -OL https://github.com/google/googletest/archive/release-${GOOGLE_TEST_VERSION}.tar.gz \
    && tar -zxvf release-${GOOGLE_TEST_VERSION}.tar.gz \
    && mkdir ./googletest-release-${GOOGLE_TEST_VERSION}/build \
	&& curl -OL https://download.pytorch.org/libtorch/nightly/cu${PYTORCH_CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}.zip \
	&& unzip libtorch-cxx11-abi-shared-with-deps-${PYTORCH_VERSION}.zip

# Google testのインストール
WORKDIR /tmp/libsrc/googletest-release-${GOOGLE_TEST_VERSION}/build
RUN cmake ..\
    && make \
    && make install
# boostのインストール
WORKDIR /tmp/libsrc/boost_${BOOST_FILE_VERSION}
RUN ./bootstrap.sh --prefix=/usr/local/ --with-libraries=program_options \
    && ./b2 install
# PyTorchのインストール
WORKDIR /tmp/libsrc/libtorch
RUN cp -v -r include/* /usr/local/include && cp -v -r lib/* /usr/local/lib && cp -v -r share/* /usr/local/share/
# インストール作業で出た一時的なファイルを削除
WORKDIR /tmp
RUN rm -rf ./libsrc/