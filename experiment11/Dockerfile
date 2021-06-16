## For docker template. It is a base file.
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

## for faster install

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

RUN echo "Installing dependencies..." && \
	apt-get -y --no-install-recommends update && \
	apt-get -y --no-install-recommends upgrade && \
	apt-get install -y --no-install-recommends \
	build-essential \
	git \
    locales \
    python3-setuptools \
    python3-pip && \
    pip3 install --upgrade pip 

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN echo "Changing python3 to python..." && \ 
    ln -s /usr/bin/python3 /usr/bin/python & \
    ln -s /usr/bin/pip3 /usr/bin/pip

## install JupyterLab
## Communication tools (openssl,wget,requests,curl)
RUN pip install requests
RUN apt-get install curl -y

## Install jupyterLab
RUN pip install jupyter --upgrade
RUN pip install jupyterlab --upgrade

RUN pip install bash_kernel
RUN python -m bash_kernel.install

## Document (pdf) tools (pandoc, texlive-xetex)
RUN apt-get install pandoc -y
RUN apt-get install texlive-xetex -y 

## End
## Create link for volume
RUN mkdir /workspace
WORKDIR /workspace

EXPOSE 8888
USER time2work
## docker build -t my-paper .

## docker run -ti --rm --gpus all --name baseline --mount type=bind,source="$(pwd)",target=/workspace -p 8888:8888 my-paper jupyter lab --ip=* --port=8888 --no-browser --notebook-dir=/workspace/ --allow-root
