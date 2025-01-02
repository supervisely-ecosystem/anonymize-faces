import os

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
        self.continue_working = True


STATE = State()
Api = sly.Api()
APP_DATA_DIR = "/sly_task_data" if sly.is_production() else "task_data"

YUNET_MODEl = None
EGOBLUR_MODEl = None

if STATE.target == Model.EGOBLUR or STATE.target == Model.BOTH:
    DEVICE = "cpu" if not cuda.is_available() else f"cuda:{cuda.current_device()}"
    if DEVICE == "cpu":
        sly.logger.warning("CUDA is unavailable on this device, falling back to using CPU for computation.")
    else:
        sly.logger.info(f"Computing on cuda:{cuda.current_device()} device")

FACE_CLASS_NAME = "face"
LP_CLASS_NAME = "license plate"
CONFIDENCE_TAG_META_NAME = "model confidence"

if sly.is_development():
    sly.logger.level = 10
