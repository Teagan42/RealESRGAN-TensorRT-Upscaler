FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/Real-ESRGAN

# Install the repository's bundled torch2trt implementation.
COPY torch2trt/setup.py ./torch2trt/setup.py
COPY torch2trt/torch2trt ./torch2trt/torch2trt
RUN python -m pip install --extra-index-url https://pypi.nvidia.com \
        tensorrt==8.6.1.post1 \
    && cd torch2trt \
    && python setup.py install

RUN python -m pip install \
    'numpy<2' \
    basicsr==1.4.2 \
    opencv-python-headless==4.10.0.84

COPY . .

CMD ["bash"]
