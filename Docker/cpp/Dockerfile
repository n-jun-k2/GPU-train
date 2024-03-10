FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV BOOST_VERSION 1.70.0
ENV BOOST_FILE_VERSION 1_70_0
ENV GOOGLE_TEST_VERSION 1.8.0
ENV PYTORCH_CUDA_VERSION 118

RUN chmod 1777 /tmp\
	&& apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false  -y update \
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
RUN curl -OL https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_FILE_VERSION}.tar.gz\
    && tar -zxvf boost_${BOOST_FILE_VERSION}.tar.gz \
    && curl -OL https://github.com/google/googletest/archive/release-${GOOGLE_TEST_VERSION}.tar.gz \
    && tar -zxvf release-${GOOGLE_TEST_VERSION}.tar.gz \
    && mkdir ./googletest-release-${GOOGLE_TEST_VERSION}/build \
	&& curl -OL https://download.pytorch.org/libtorch/cu${PYTORCH_CUDA_VERSION}/libtorch-win-shared-with-deps-2.0.1%2Bcu${PYTORCH_CUDA_VERSION}.zip \
	&& unzip libtorch-win-shared-with-deps-2.0.1%2Bcu${PYTORCH_CUDA_VERSION}.zip

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
RUN cp -v -r include/* /usr/local/include \
	&& cp -v -r lib/* /usr/local/lib \
	&& cp -v -r share/* /usr/local/share/
# インストール作業で出た一時的なファイルを削除
WORKDIR /tmp
RUN rm -rf ./libsrc/