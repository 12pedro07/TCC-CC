ARG CUDA="11.3.1"
ARG CUDNN="8"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev wget

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

# Setup PATH to install python with miniconda
ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name tcc-py37 python=3.7.9 \
 && /miniconda/bin/conda clean -ya

# Setup environment variables
ENV CONDA_DEFAULT_ENV=tcc-py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install insightface and dependencies
RUN apt-get install -y pandoc
RUN pip install onnxruntime-gpu Cython pypandoc pandoc mxnet
ENV PATH=${CONDA_PREFIX}/lib/python3.7/site-packages/pandoc:$PATH
RUN pip install insightface

# Install opencv
RUN pip install opencv-python==4.3.0.36 opencv-contrib-python==4.3.0.36 opencv-python-headless==4.5.1.48

# Install other usefull packages
RUN pip install numpy pathlib

# enable cuda
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

# setup project
WORKDIR /src
