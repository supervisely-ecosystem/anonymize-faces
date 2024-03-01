FROM supervisely/base-py-sdk:6.73.35

COPY requirements.txt /workdir/requirements.txt

RUN pip3 install -r /workdir/requirements.txt

COPY . /workdir

COPY models /sly_task_data/models