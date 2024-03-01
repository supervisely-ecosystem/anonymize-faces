#!/bin/bash

if [[ -z "${TAG}" ]]; then
  echo "TAG is not set. Please set TAG environment variable"
  exit 1
fi

python3 docker/download_models.py
docker build . -t supervisely/anonymize:${TAG} -f docker/Dockerfile
docker push supervisely/anonymize:${TAG}
