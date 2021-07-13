FROM nvidia/vulkan:1.1.121-cuda-10.1--ubuntu18.04

ENV CMAKE_VERSION 3.20.5
ENV CMAKE_MAJOR_VERSION 3.20
ENV BOOST_VERSION 1.70.0
ENV BOOST_FILE_VERSION 1_70_0
ENV GOOGLE_TEST_VERSION 1.8.0
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV NVIDIA_DRIVER_VERSION 430.26
ENV VULKAN_LIBRARY_PATH libGLX_nvidia.so.0
# ENV VULKAN_LIBRARY_PATH libGLX.so.0

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TERM=linux apt-get install --no-install-recommends -q -y \ 
        curl  \
        ca-certificates \
        build-essential \
		zlib1g-dev \
		libffi-dev \
        libx11-xcb-dev \
        libxkbcommon-dev \
        libmirclient-dev \
        libwayland-dev \
        libxrandr-dev \
		zip \ 
		unzip \
        wget \
        git \
        ninja-build \
        vulkan-utils libvulkan-dev libvulkan1 \
        libnvidia-gl-390 \
        libxcb1-dev \
        libx11-dev \
    && apt-get clean \
    && rm -r /var/lib/apt/lists/*

WORKDIR /tmp/src

RUN wget -q --show-progress --progress=bar:force:noscroll http://us.download.nvidia.com/XFree86/Linux-x86_64/$NVIDIA_DRIVER_VERSION/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run -O /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run \
    && cd /tmp \
    && sh NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run --extract-only \
    && cp /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION/libnvidia-cbl.so.$NVIDIA_DRIVER_VERSION /usr/lib/x86_64-linux-gnu/ \
    && cp /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION/libnvidia-glvkspirv.so.$NVIDIA_DRIVER_VERSION /usr/lib/x86_64-linux-gnu/ \
    && cp /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION/libnvidia-rtcore.so.$NVIDIA_DRIVER_VERSION /usr/lib/x86_64-linux-gnu/ \
    && rm -rf /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION /tmp/NVIDIA-Linux-x86_64-$NVIDIA_DRIVER_VERSION.run

RUN VULKAN_API_VERSION=`dpkg -s libvulkan1 | grep -oP 'Version: [0-9|\.]+' | grep -oP '[0-9|\.]+'` \
    && mkdir -p /etc/vulkan/icd.d/ \
    && echo \
	"{\
		\"file_format_version\" : \"1.0.0\",\
		\"ICD\": {\
			\"library_path\": \"${VULKAN_LIBRARY_PATH}\",\
			\"api_version\" : \"${VULKAN_API_VERSION}\"\
		}\
	}" > /etc/vulkan/icd.d/nvidia_icd.json \
	&& rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    && tar -xvf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    && curl -OL https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_FILE_VERSION}.tar.gz\
    && tar -zxvf boost_${BOOST_FILE_VERSION}.tar.gz \
    && curl -OL https://github.com/google/googletest/archive/release-${GOOGLE_TEST_VERSION}.tar.gz \
    && tar -zxvf release-${GOOGLE_TEST_VERSION}.tar.gz \
    && mkdir ./googletest-release-${GOOGLE_TEST_VERSION}/build

# cmakeのインストール
RUN cp cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake /usr/local/bin \
    && mkdir -p /usr/local/share/cmake-${CMAKE_MAJOR_VERSION} \
    && cp -r cmake-${CMAKE_VERSION}-linux-x86_64/share/cmake-${CMAKE_MAJOR_VERSION}/* /usr/local/share/cmake-${CMAKE_MAJOR_VERSION}

# Google testのインストール
WORKDIR /tmp/src/googletest-release-${GOOGLE_TEST_VERSION}/build
RUN cmake ..\
    && make \
    && make install
# boostのインストール
WORKDIR /tmp/src/boost_${BOOST_FILE_VERSION}
RUN ./bootstrap.sh --prefix=/usr/local/ --with-libraries=program_options \
    && ./b2 install

WORKDIR /tmp
RUN rm -rf /tmp/src
#ldconfig -p | grep -i libGL