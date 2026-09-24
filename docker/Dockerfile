# Model weights are taken from the previous image: the EgoBlur download link has expired
FROM supervisely/anonymize:1.2.3 AS models

# torch builds with RTX 50xx (sm_120) kernels need Python 3.9+, which base-py-sdk (Python 3.8) does not have
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libmagic1 \
    libgeos-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir opencv-python==4.9.0.80 supervisely==6.73.214

COPY . /workdir

COPY --from=models /sly_task_data/models/ego_blur_lp.jit /sly_task_data/models/face_detection_yunet_2023mar.onnx /sly_task_data/models/

LABEL python_sdk_version=6.73.214
