# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 as builder

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN apt-get update && apt-get install -y wget \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.9 get-pip.py \
    && rm get-pip.py

# Update alternatives to prioritize Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9

# Update the symbolic link for python to point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Verify installation
RUN python --version
RUN python3 --version
RUN pip --version
# copy requirements file and and install
COPY ./requirements.txt /opt/
RUN pip3 install --no-cache-dir -r /opt/requirements.txt
# copy src code into image and chmod scripts
COPY src ./opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh
COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"
# Set working directory
WORKDIR /opt/src
# set python variables and path
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
ENV TORCH_HOME="/opt"
ENV MPLCONFIGDIR="/opt"

RUN chown -R 1000:1000 /opt

RUN chmod -R 777 /opt

# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]