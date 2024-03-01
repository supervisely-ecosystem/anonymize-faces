import os
from pathlib import Path

from src.utils import download_models

if __name__ == "__main__":
    download_models(Path(os.getcwd(), "/models"))
