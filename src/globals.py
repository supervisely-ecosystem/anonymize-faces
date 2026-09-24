import os
import re

import supervisely as sly
from dotenv import load_dotenv
from torch import cuda

if sly.is_development():
    # * For convinient development, has no effect in the production.
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))


class Shape:
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


class Method:
    BLUR = "blur"
    SOLID = "solid"

class Model:
    YUNET = "yunet"
    EGOBLUR = "egoblur"
    BOTH = "both"


AVAILABLE_SHAPES = [Shape.RECTANGLE, Shape.ELLIPSE]
AVAILABLE_METHODS = [Method.BLUR, Method.SOLID]


class ModalState:
    """Modal state"""

    SHAPE = "modal.state.Shape"
    METHOD = "modal.state.Method"
    SAVE_DETECTIONS = "modal.state.SaveDetections"
    ANONYMIZE = "modal.state.Anonymize"
    TARGET = "modal.state.Target"
    RESIZE_VIDEOS = "modal.state.ResizeVideos"
    RESIZE_PERCENTAGE = "modal.state.ResizePercentage"
    FACE_DETECTION_SCALE = "modal.state.FaceDetectionScale"

    def shape(self):
        return os.environ.get(self.SHAPE, Shape.RECTANGLE)

    def method(self):
        return os.environ.get(self.METHOD, Method.BLUR)

    def anonymize(self):
        val = os.environ.get(self.ANONYMIZE, True)
        return val in ("True", "true", "1", True)

    def save_detections(self):
        val = os.environ.get(self.SAVE_DETECTIONS, True)
        return val in ("True", "true", "1", True)

    def threshold(self):
        return float(os.environ.get("modal.state.Threshold", 0.55))

    def target(self):
        return os.environ.get(self.TARGET, Model.BOTH)

    def resize_videos(self):
        val = os.environ.get(self.RESIZE_VIDEOS, False)
        return val in ("True", "true", "1", True)

    def resize_percentage(self):
        return float(os.environ.get(self.RESIZE_PERCENTAGE, 100))

    def face_detection_scale(self):
        try:
            val = float(os.environ.get(self.FACE_DETECTION_SCALE, 100))
        except ValueError:
            val = 100
        return min(max(val, 1), 100)


class State:
    """App state"""

    def __init__(self):
        self.selected_team = sly.env.team_id()
        self.selected_workspace = sly.env.workspace_id()
        self.selected_project = sly.env.project_id()
        self.selected_dataset = sly.env.dataset_id(raise_not_found=False)
        self.obfuscate_shape = ModalState().shape()
        self.obfuscate_method = ModalState().method()
        self.should_anonymize = ModalState().anonymize()
        self.should_save_detections = ModalState().save_detections()
        self.threshold = ModalState().threshold()
        self.target = ModalState().target()
        self.resize_videos = ModalState().resize_videos()
        self.resize_percentage = ModalState().resize_percentage()
        self.face_detection_scale = ModalState().face_detection_scale()
        self.continue_working = True


STATE = State()
Api = sly.Api()
APP_DATA_DIR = "/sly_task_data" if sly.is_production() else "task_data"

YUNET_MODEl = None
EGOBLUR_MODEl = None


def _cuda_supports_device() -> bool:
    """Whether this torch build has kernels the current GPU can run"""
    major, minor = cuda.get_device_capability()
    for arch in cuda.get_arch_list():
        match = re.fullmatch(r"(sm|compute)_(\d+)(\d)[a-z]?", arch)
        if match is None:
            continue
        kind, arch_major, arch_minor = match[1], int(match[2]), int(match[3])
        # a binary runs on a newer GPU of the same major, PTX on any newer GPU
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


if STATE.target == Model.EGOBLUR or STATE.target == Model.BOTH:
    DEVICE = "cpu"
    if cuda.is_available():
        if _cuda_supports_device():
            DEVICE = f"cuda:{cuda.current_device()}"
        else:
            sly.logger.warning(
                f"{cuda.get_device_name()} is not supported by this build of torch "
                f"({', '.join(cuda.get_arch_list())})."
            )
    if DEVICE == "cpu":
        sly.logger.warning(
            "CUDA is unavailable, license plate detection will run on CPU and be very slow "
            "(seconds per frame). Run the app on an agent with a GPU, or select Faces only. "
            "A GPU agent also falls back to CPU if its NVIDIA driver does not support CUDA 12."
        )
    else:
        sly.logger.info(f"Computing on cuda:{cuda.current_device()} device")

FACE_CLASS_NAME = "face"
LP_CLASS_NAME = "license plate"
CONFIDENCE_TAG_META_NAME = "model confidence"

if sly.is_development():
    sly.logger.level = 10
