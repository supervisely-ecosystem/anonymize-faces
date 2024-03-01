import os
from pathlib import Path
from zipfile import ZipFile

import requests


def download_yunet_model(path: str = None):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    model_path = Path(path)
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=60)
        with open(model_path, "wb") as f:
            f.write(r.content)
    return model_path.absolute()


def download_egoblur_model(path: str = None):
    url = "https://scontent.fopo5-1.fna.fbcdn.net/m1/v/t6/An9sSpp_UpJ9wK4iapy8E1sWowJGvE3s-_npVcbow_FqLT4OJ0kiLsLOEnIUMC290kM3mfain4-Oomukg7ROXPYZr7YVpc8dJo-VYdOyneJ7HQNa8oi35HOE-H4yJ50wcKXc5eGeIg.zip/ego_blur_lp.zip?sdl=1&ccb=10-5&oh=00_AfAMHgC_-Bb7Bi3xA6rdCK5a8bTrzmQPTnL4vUt-gIN9zQ&oe=66073B3E&_nc_sid=5cb19f"
    file_name_zip = "ego_blur_lp.zip"
    model_path = Path(path)
    if model_path.is_file():
        return model_path.absolute()

    model_path_zip = model_path.with_name(file_name_zip)
    model_path_zip.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    with open(model_path_zip, "wb") as f:
        f.write(r.content)
    with ZipFile(model_path_zip.absolute(), "r") as zip_ref:
        zip_ref.extractall(model_path_zip.parent)

    return model_path.absolute()


def download_models(dir: str = None):
    download_yunet_model(Path(dir, "face_detection_yunet_2023mar.onnx"))
    download_egoblur_model(Path(dir, "ego_blur_lp.jit"))


if __name__ == "__main__":
    download_models(Path(os.getcwd(), "models"))
