FROM ubuntu:latest
MAINTAINER Valentin Kuznetsov vkuznet@gmail.com

# args to use
ARG tag

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  git curl openssh-client ca-certificates python3 python3-pip python3-venv && \
  update-ca-certificates && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# build procedure
ENV WDIR=/data
WORKDIR $WDIR
RUN mkdir /build
RUN git clone https://github.com/CHESSComputing/ChessAnalysisPipeline.git
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN cd ChessAnalysisPipeline && \
pip3 install setuptools && \
pip3 install --upgrade build && \
pip3 install --no-cache-dir -r requirements.txt && \
sed -i "s/PACKAGE_VERSION/${tag}/" setup.py  && \
sed -i "s/PACKAGE_VERSION/${tag}/" CHAP/__init__.py && \
python3 -m build
RUN pip3 install ChessAnalysisPipeline/dist/chessanalysispipeline-0.0.9-py3-none-any.whl
CMD ["/opt/venv/bin/CHAP"]
