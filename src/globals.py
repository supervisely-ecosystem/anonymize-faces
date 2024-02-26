import os

import supervisely as sly
from dotenv import load_dotenv

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


AVAILABLE_SHAPES = [Shape.RECTANGLE, Shape.ELLIPSE]
AVAILABLE_METHODS = [Method.BLUR, Method.SOLID]


class ModalState:
    """Modal state"""

    SHAPE = "modal.state.Shape"
    METHOD = "modal.state.Method"
    SAVE_DETECTIONS = "modal.state.SaveDetections"
    ANONYMIZE = "modal.state.Anonymize"

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
        self.continue_working = True


STATE = State()
Api = sly.Api()
APP_DATA_DIR = "/sly_task_data" if sly.is_production() else "task_data"

YUNET_MODEl = None
FACE_CLASS_NAME = "face"
CONFIDENCE_TAG_META_NAME = "model confidence"
