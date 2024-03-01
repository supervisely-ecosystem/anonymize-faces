#!/bin/bash

if [[ -z "${TAG}" ]]; then
  echo "TAG is not set. Please set TAG environment variable"
  exit 1
fi

python3 download_models.py
docker build . -t supervisely/anonymize:${TAG}
docker push supervisely/anonymize:${TAG}
