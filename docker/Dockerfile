FROM supervisely/base-py-sdk:6.73.214

RUN pip install opencv-python==4.9.0.80 torch==2.2.1 torchvision==0.17.1

COPY . /workdir

COPY models /sly_task_data/models

LABEL python_sdk_version=6.73.214
