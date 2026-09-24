import json
import os
import subprocess
from pathlib import Path
import time
from typing import Callable, List, Optional, Dict
from zipfile import ZipFile, is_zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import supervisely as sly
import torch
import torchvision

from supervisely.video.video import get_info

import globals as g


def get_obj_class_names():
    if g.STATE.target == g.Model.BOTH:
        return [g.FACE_CLASS_NAME, g.LP_CLASS_NAME]
    elif g.STATE.target == g.Model.YUNET:
        return [g.FACE_CLASS_NAME]
    elif g.STATE.target == g.Model.EGOBLUR:
        return [g.LP_CLASS_NAME]


def updated_project_meta(project_meta: sly.ProjectMeta) -> sly.ProjectMeta:
    """Update project meta to include anonymized objects"""
    obj_classes = project_meta.obj_classes

    obj_class_names = get_obj_class_names()
    for obj_class_name in obj_class_names:
        if not obj_classes.has_key(obj_class_name):
            obj_classes = obj_classes.add(sly.ObjClass(obj_class_name, sly.Rectangle))
    project_meta = project_meta.clone(obj_classes=obj_classes)
    tag_metas = project_meta.tag_metas
    if not g.CONFIDENCE_TAG_META_NAME in tag_metas:
        tag_metas = tag_metas.add(
            sly.TagMeta(
                g.CONFIDENCE_TAG_META_NAME,
                sly.TagValueType.ANY_NUMBER,
                applicable_to=sly.TagApplicableTo.OBJECTS_ONLY,
            )
        )
        project_meta = project_meta.clone(tag_metas=tag_metas)
    return project_meta


def create_dst_project(src_project: sly.ProjectInfo) -> sly.ProjectInfo:
    """Create a new project for anonymized images"""
    src_project_meta = sly.ProjectMeta.from_json(g.Api.project.get_meta(src_project.id))
    description = (
        f'Created from project "{src_project.name}" (id: {src_project.id})\n'
        f"{src_project.description}"
    )
    dst_project = g.Api.project.create(
        workspace_id=src_project.workspace_id,
        name=f"{src_project.name}_anonymized",
        description=description,
        type=src_project.type,
        change_name_if_conflict=True,
    )
    dst_project_meta = (
        updated_project_meta(src_project_meta)
        if g.STATE.should_save_detections
        else src_project_meta
    )
    g.Api.project.update_meta(dst_project.id, dst_project_meta)
    dst_custom_data = {
        **src_project.custom_data,
        "anonymized_objects": True,
        "src_project_id": src_project.id,
    }
    g.Api.project.update_custom_data(dst_project.id, dst_custom_data)
    copy_project_settings(src_project.id, dst_project.id)
    return dst_project


def copy_project_settings(src_project_id: int, dst_project_id: int):
    """Give the anonymized project the source's settings, labeling interface included.

    Without this a video project on the telemetry interface comes out on the default one, and the
    restored GPS track has no map to show it on. Raw API calls on purpose: the SDK in this image
    validates the interface against its own list and rejects values newer than itself.
    """
    try:
        settings = g.Api.post("projects.info", {"id": src_project_id}).json().get("settings")
        if settings:
            g.Api.post("projects.settings.update", {"id": dst_project_id, "settings": settings})
    except Exception as e:
        sly.logger.warning(f"Could not copy project settings to the anonymized project: {e}")


def create_dst_dataset(
    src_dataset: sly.DatasetInfo, dst_project: sly.ProjectInfo, parent_id: Optional[int] = None
) -> sly.DatasetInfo:
    """Create a new dataset for anonymized images"""
    description = (
        f'Created from dataset "{src_dataset.name}" (id: {src_dataset.id})\n'
        f"{src_dataset.description}"
    )
    dst_dataset = g.Api.dataset.create(
        project_id=dst_project.id,
        name=src_dataset.name,
        description=description,
        change_name_if_conflict=False,
        parent_id=parent_id,
    )

    return dst_dataset


def download_yunet_model(path: str = None):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    file_name = "face_detection_yunet_2023mar.onnx"
    if path is None:
        model_path = Path(g.APP_DATA_DIR, "models", file_name)
    else:
        model_path = Path(path)
    if not model_path.exists():
        sly.logger.info("Model for face detection not found, downloading it...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=60)
        with open(model_path, "wb") as f:
            f.write(r.content)
    return model_path.absolute()


def get_yunet_model():
    if g.YUNET_MODEl is None:
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
        model_path = str(download_yunet_model())

        yunet = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),  # default, can be changed
            score_threshold=g.STATE.threshold,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=backend_id,
            target_id=target_id,
        )

        g.YUNET_MODEl = yunet
    return g.YUNET_MODEl


def detect_faces_yunet(img: np.ndarray) -> np.ndarray:
    model = get_yunet_model()
    # YuNet runs on CPU, where a 4K frame costs over a second: detecting on a downscaled copy is
    # much faster, and the boxes are scaled back to the original frame
    scale = g.STATE.face_detection_scale / 100
    if scale < 1:
        width = max(1, round(img.shape[1] * scale))
        height = max(1, round(img.shape[0] * scale))
        scale_x, scale_y = width / img.shape[1], height / img.shape[0]
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    else:
        scale_x = scale_y = 1
    model.setInputSize((img.shape[1], img.shape[0]))
    _, faces = model.detect(img)
    if faces is None:
        return []

    def _convert(coord: str, scale: float):
        return max(0, int(coord / scale))

    res = []
    for face in faces:
        conf = float(face[-1])
        scales = (scale_x, scale_y, scale_x, scale_y)
        face_coords = [_convert(x, s) for x, s in zip(face[:4], scales)]
        res.append([*face_coords, conf])
    return res


def convert_bbox_to_coco(box: List) -> List:
    x1, y1, x2, y2 = box
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return [int(x), int(y), int(w), int(h)]


def download_egoblur_model(path: str = None):
    url = "https://scontent.fopo5-1.fna.fbcdn.net/m1/v/t6/An9sSpp_UpJ9wK4iapy8E1sWowJGvE3s-_npVcbow_FqLT4OJ0kiLsLOEnIUMC290kM3mfain4-Oomukg7ROXPYZr7YVpc8dJo-VYdOyneJ7HQNa8oi35HOE-H4yJ50wcKXc5eGeIg.zip/ego_blur_lp.zip?_nc_oc=Adg3NONIMTL7HzrjXFQz_n8D2sgRroTMlk1n0tk3I9rxOGcITYuDKlDNATEYgv_n-UOg0J4zAWUJ5-AIvhOva2en&sdl=1&ccb=10-5&oh=00_AYCLGGhU0TBrn1D45VbLzfldOrjdMG91xWsCZvxGkn-6ow&oe=679DF1FE&_nc_sid=5cb19f"
    file_name_zip = "ego_blur_lp.zip"
    file_name = "ego_blur_lp.jit"
    if path is None:
        model_path = Path(g.APP_DATA_DIR, "models", file_name)
    else:
        model_path = Path(path)
    if model_path.is_file():
        return model_path.absolute()

    sly.logger.info("Model for license plate detection not found, downloading it...")
    model_path_zip = model_path.with_name(file_name_zip)
    model_path_zip.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    with open(model_path_zip, "wb") as f:
        f.write(r.content)
    if not is_zipfile(model_path_zip.absolute()):
        raise ValueError("Downloaded file is not a valid zip file")
    with ZipFile(model_path_zip.absolute(), "r") as zip_ref:
        zip_ref.extractall(model_path_zip.parent.absolute())

    return model_path.absolute()


def get_lp_egoblur():
    if g.EGOBLUR_MODEl is None:
        lp_model_path = str(download_egoblur_model())
        lp_detector = torch.jit.load(lp_model_path, map_location="cpu").to(g.DEVICE)
        lp_detector.eval()

        g.EGOBLUR_MODEl = lp_detector
    return g.EGOBLUR_MODEl


def detect_lp_egoblur(
    image: np.ndarray,
):
    model_score_threshold = g.STATE.threshold
    nms_iou_threshold = 0.3
    detector = get_lp_egoblur()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_transposed = np.transpose(image, (2, 0, 1))
    image_tensor = torch.from_numpy(image_transposed).to(device=g.DEVICE)

    # fp16 on GPU is much faster and moves the boxes by a fraction of a pixel; CPU stays in fp32
    use_fp16 = g.DEVICE.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims
    boxes, scores = boxes.float(), scores.float()

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]

    res = [
        convert_bbox_to_coco(box.tolist()) + [float(score)]
        for box, score in zip(boxes, scores)
        if score > model_score_threshold
    ]
    return res


DETECTORS_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def run_detectors(detectors: List[Callable], img: np.ndarray) -> List[tuple]:
    """Run every detector on the image and return (detector, detections, seconds) in their order.

    With both models selected they run side by side: YuNet on CPU while EgoBlur is on the GPU.
    cv2 and torch release the GIL, so two threads are enough to overlap them.
    """

    def _detect(detector):
        t = time.time()
        dets = detector(img)
        return detector, dets, round(time.time() - t, 3)

    if len(detectors) == 1:
        return [_detect(detectors[0])]
    futures = [DETECTORS_EXECUTOR.submit(_detect, detector) for detector in detectors]
    return [future.result() for future in futures]


def _get_rectangles_mask(size, objects):
    mask = np.zeros(size, dtype=np.uint8)
    for x, y, w, h in objects:
        mask[y : y + h, x : x + w] = 1
    return mask


def blur_objects(img, objects, shape: str):
    for x, y, w, h in objects:
        if shape == g.Shape.RECTANGLE:
            img[y : y + h, x : x + w] = cv2.blur(img[y : y + h, x : x + w], (h // 2, w // 2))
        elif shape == g.Shape.ELLIPSE:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask = cv2.ellipse(
                mask,
                (x + w // 2, y + h // 2),
                (w // 2, h // 2),
                0,
                0,
                360,
                color=1,
                thickness=-1,
            )
            img_with_blurred_rect = img.copy()
            img_with_blurred_rect[y : y + h, x : x + w] = cv2.blur(
                img[y : y + h, x : x + w], (h // 2, w // 2)
            )
            img = np.where(mask[:, :, None] == 1, img_with_blurred_rect, img)
    return img


def obfuscate_objects(img: np.ndarray, objects: List, shape: str, method: str) -> np.ndarray:
    if shape not in g.AVAILABLE_SHAPES:
        raise ValueError(f"Invalid shape: {shape}")
    if method not in g.AVAILABLE_METHODS:
        raise ValueError(f"Invalid method: {method}")

    if method == g.Method.BLUR:
        return blur_objects(img, objects, shape)
    else:
        if shape == g.Shape.RECTANGLE:
            mask = _get_rectangles_mask(img.shape[:2], objects)
        elif shape == g.Shape.ELLIPSE:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for x, y, w, h in objects:
                mask = cv2.ellipse(
                    mask,
                    (x + w // 2, y + h // 2),
                    (w // 2, h // 2),
                    0,
                    0,
                    360,
                    color=1,
                    thickness=-1,
                )
        return np.where(mask[:, :, None] == 1, 0, img)


def update_annotation(dets, class_name, annotation: sly.Annotation, project_meta: sly.ProjectMeta):
    conf_tag_meta = project_meta.get_tag_meta(g.CONFIDENCE_TAG_META_NAME)

    obj_class = project_meta.get_obj_class(class_name)
    labels = []
    for det in dets:
        x, y, w, h, conf = det

        label = sly.Label(
            sly.Rectangle(y, x, y + h, x + w),
            obj_class=obj_class,
            tags=[sly.Tag(conf_tag_meta, conf)],
        )
        labels.append(label)
    return annotation.add_labels(labels)


def coords_from_rectangle(rect: sly.Rectangle):
    return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)


def objects_from_annotation(ann: sly.Annotation):
    return [
        coords_from_rectangle(label.geometry)
        for label in ann.labels
        if (label.obj_class.name == g.FACE_CLASS_NAME and isinstance(label.geometry, sly.Rectangle))
    ]


def filter_objects(objects: List):
    objects = [tuple(obj) for obj in objects]
    objects = set(objects)
    return list(objects)


def fix_codec(input_video_path):
    output_video_path = os.path.splitext(input_video_path)[0] + "_h264" + ".mp4"

    # read video meta_data
    need_video_transc = False
    try:
        video_meta = get_info(input_video_path)
        for stream in video_meta["streams"]:
            codec_type = stream["codecType"]
            if codec_type not in ["video", "audio"]:
                continue
            codec_name = stream["codecName"]
            if codec_type == "video" and codec_name != "h264":
                need_video_transc = True
    except:
        need_video_transc = True

    # convert videos
    if need_video_transc:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                f"{input_video_path}",
                "-c:v",
                f"libx264",
                "-c:a",
                f"copy",
                f"{output_video_path}",
            ],
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        sly.fs.silent_remove(input_video_path)
        os.rename(output_video_path, input_video_path)


# Containers that can carry copied-through data streams. Anything else is left untouched.
REMUXABLE_CONTAINERS = (".mp4", ".mov")
# Timecode tracks often refuse to remux into a re-encoded file and carry no useful payload here.
SKIP_DATA_STREAM_TAGS = ("tmcd",)
# Tolerance when comparing source and anonymized video duration, in seconds.
DURATION_TOLERANCE = 0.5


def _probe_streams(video_path: str) -> List[Dict]:
    """Return ffprobe stream descriptors for a video file, or an empty list if probing fails"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", video_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sly.logger.warning(f"ffprobe failed for {video_path}: {result.stderr}")
        return []
    try:
        return json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        sly.logger.warning(f"Could not parse ffprobe output for {video_path}")
        return []


def _video_duration(streams: List[Dict]) -> Optional[float]:
    for stream in streams:
        if stream.get("codec_type") == "video" and stream.get("duration") is not None:
            return float(stream["duration"])
    return None


def _streams_to_restore(streams: List[Dict]) -> List[int]:
    """Indices of the source streams that the frame-by-frame rewrite drops: telemetry and audio"""
    indices = []
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type not in ("data", "audio"):
            continue
        if codec_type == "data" and stream.get("codec_tag_string") in SKIP_DATA_STREAM_TAGS:
            continue
        indices.append(stream["index"])
    return indices


def restore_source_streams(anonymized_video_path: str, source_video_path: str):
    """Copy non-video streams and container metadata from the source video into the anonymized one.

    The anonymized video is rebuilt frame by frame with cv2, so it ends up holding a video stream
    and nothing else: data tracks such as the GoPro gpmd telemetry (GPS, accelerometer, gyroscope)
    and any audio are lost, together with container-level metadata. They are stream-copied back
    here, which leaves the anonymized pixels untouched. A no-op when the source has nothing extra.
    """
    if os.path.splitext(anonymized_video_path)[1].lower() not in REMUXABLE_CONTAINERS:
        sly.logger.debug("Output container does not support stream copy, skipping restore")
        return
    if not os.path.isfile(source_video_path):
        sly.logger.warning("Source video is unavailable, cannot restore its streams")
        return

    source_streams = _probe_streams(source_video_path)
    indices = _streams_to_restore(source_streams)
    if not indices:
        sly.logger.debug("Source video has no data or audio streams to restore")
        return

    # The restored streams keep the timestamps they had in the source, so they only line up if
    # the anonymized video still covers the same span of time.
    source_duration = _video_duration(source_streams)
    output_duration = _video_duration(_probe_streams(anonymized_video_path))
    if source_duration is None or output_duration is None:
        sly.logger.warning("Could not compare video durations, skipping stream restore")
        return
    if abs(source_duration - output_duration) > DURATION_TOLERANCE:
        sly.logger.warning(
            "Anonymized video duration differs from the source, skipping stream restore "
            "to avoid desynchronized metadata",
            extra={"source_duration": source_duration, "output_duration": output_duration},
        )
        return

    base, ext = os.path.splitext(anonymized_video_path)
    restored_path = f"{base}_restored{ext}"
    cmd = ["ffmpeg", "-y", "-i", anonymized_video_path, "-i", source_video_path, "-map", "0:v:0"]
    for index in indices:
        cmd.extend(["-map", f"1:{index}"])
    # -copy_unknown is required: without it ffmpeg drops streams it cannot decode, gpmd included
    cmd.extend(["-c", "copy", "-copy_unknown", "-map_metadata", "1", restored_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sly.fs.silent_remove(restored_path)
        sly.logger.warning(
            f"Could not restore source streams, uploading video without them: {result.stderr}"
        )
        return

    sly.fs.silent_remove(anonymized_video_path)
    os.rename(restored_path, anonymized_video_path)
    sly.logger.debug(f"Restored {len(indices)} stream(s) from the source video")


def run_images(
    src_dataset: sly.DatasetInfo,
    dst_dataset: sly.DatasetInfo,
    dst_project_meta: sly.ProjectMeta,
    detectors: List[Callable],
    progress,
):
    download_executor = ThreadPoolExecutor(max_workers=10)
    upload_executor = ThreadPoolExecutor(max_workers=4)
    try:
        for batch in g.Api.image.get_list_generator(src_dataset.id, batch_size=50):
            img_cache = {}
            is_downloading = {}

            sly.logger.debug(f"Processing batch of {len(batch)} images")
            batch_timings = {}
            t = time.time()
            ann_infos = g.Api.annotation.download_batch(src_dataset.id, [img.id for img in batch])
            anns_dict = {
                ann_info.image_id: sly.Annotation.from_json(
                    ann_info.annotation, project_meta=dst_project_meta
                )
                for ann_info in ann_infos
            }
            batch_timings["download annotations"] = round(time.time() - t, 3)

            def _download_image(image_id):
                if image_id in is_downloading and is_downloading[image_id]:
                    sly.logger.debug(
                        f"Waiting for image [{image_id}] to download...",
                        extra={"image_id": image_id},
                    )
                    while is_downloading[image_id]:
                        time.sleep(0.1)
                if image_id not in img_cache:
                    is_downloading[image_id] = True
                    img_cache[image_id] = g.Api.image.download_np(image_id)
                    is_downloading[image_id] = False
                return img_cache[image_id]

            for image_info in batch:
                download_executor.submit(_download_image, image_info.id)

            dst_nps = []
            for image_info in batch:
                timings = {"image_id": image_info.id}
                t = time.time()
                img = _download_image(image_info.id)
                timings["download image"] = round(time.time() - t, 3)
                for detector, dets, detection_time in run_detectors(detectors, img):
                    timings.setdefault(detector.__name__, {})["detection"] = detection_time
                    class_name = (
                        g.FACE_CLASS_NAME
                        if detector.__name__ == "detect_faces_yunet"
                        else g.LP_CLASS_NAME
                    )
                    t = time.time()
                    if g.STATE.should_save_detections:
                        anns_dict[image_info.id] = update_annotation(
                            dets, class_name, anns_dict[image_info.id], dst_project_meta
                        )
                    timings[detector.__name__]["update annotation"] = round(time.time() - t, 3)
                    t = time.time()
                    if g.STATE.should_anonymize:
                        objects = [det[:4] for det in dets]
                        objects.extend(objects_from_annotation(anns_dict[image_info.id]))
                        img = obfuscate_objects(
                            img,
                            filter_objects(objects),
                            g.STATE.obfuscate_shape,
                            g.STATE.obfuscate_method,
                        )
                    timings[detector.__name__]["obfuscate objects"] = round(time.time() - t, 3)
                sly.logger.debug(
                    f"Processed image {image_info.id} with detectors: {[detector.__name__ for detector in detectors]}",
                    extra={"timings": timings},
                )
                batch_timings.setdefault("images", []).append(timings)
                dst_nps.append(img)

            t = time.time()
            dst_names = [image_info.name for image_info in batch]
            dst_img_metas = [
                {
                    **image_info.meta,
                    "anonymized_objects": g.STATE.should_anonymize,
                    "src_image_id": image_info.id,
                }
                for image_info in batch
            ]

            def _upload_images(dst_dataset_id, dst_names, dst_nps, dst_img_metas):
                t = time.time()
                dst_images = g.Api.image.upload_nps(
                    dst_dataset_id, dst_names, dst_nps, metas=dst_img_metas
                )
                g.Api.annotation.upload_anns(
                    [img.id for img in dst_images], [anns_dict[img.id] for img in batch]
                )
                progress.update(len(dst_images))
                sly.logger.debug(
                    "Uploaded images and annotations",
                    extra={"timings": {"upload": time.time() - t}},
                )

            sly.logger.debug("Submitting images for upload")
            upload_executor.submit(
                _upload_images, dst_dataset.id, dst_names, dst_nps, dst_img_metas
            )

            batch_avg_yunet_processing_time = (
                round(
                    sum(
                        image_timings.get("detect_faces_yunet", {}).get("detection", 0)
                        for image_timings in batch_timings.get("images", [])
                    )
                    / len(batch),
                    3,
                )
                if len(batch) > 0
                else "N/A"
            )
            batch_avg_lp_processing_time = (
                round(
                    sum(
                        image_timings.get("detect_lp_egoblur", {}).get("detection", 0)
                        for image_timings in batch_timings.get("images", [])
                    )
                    / len(batch),
                    3,
                )
                if len(batch) > 0
                else "N/A"
            )
            sly.logger.debug(
                f"Processed batch of {len(batch)} images",
                extra={
                    "avg_yunet_det_timing": batch_avg_yunet_processing_time,
                    "avg_lp_det_timing": batch_avg_lp_processing_time,
                    "timings": batch_timings,
                },
            )
        download_executor.shutdown(wait=True)
        upload_executor.shutdown(wait=True)
    finally:
        import sys

        if sys.version_info >= (3, 9):
            download_executor.shutdown(wait=False, cancel_futures=True)
            upload_executor.shutdown(wait=False, cancel_futures=True)
        else:
            download_executor.shutdown(wait=False)
            upload_executor.shutdown(wait=False)


def video_only_copy(video_path: str) -> str:
    """A stream copy of just the first video stream, or the path itself when that is all it has."""
    streams = _probe_streams(video_path)
    if streams and all(st.get("codec_type") == "video" for st in streams):
        return video_path
    base, ext = os.path.splitext(video_path)
    dst = f"{base}_video_only{ext}"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-map", "0:v:0", "-c", "copy", dst]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sly.logger.warning(f"Could not extract the video stream, decoding the original: {result.stderr}")
        sly.fs.silent_remove(dst)
        return video_path
    return dst


def resize_video(input_video_path: str, percentage: int) -> str:
    """Write a resized copy of the video and return its path. The input is left in place"""
    output_video_path = os.path.splitext(input_video_path)[0] + "_resized.mp4"

    # Get original video dimensions
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Calculate new dimensions
    new_width = int(width * (percentage / 100))
    new_height = int(height * (percentage / 100))
    sly.logger.debug(f"Resizing video: {width}x{height} -> {new_width}x{new_height}")

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-vf",
            f"scale={new_width}:{new_height}",
            output_video_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"An ffmpeg error occurred while resizing video: {result.stderr}")

    return output_video_path


def run_videos(
    src_dataset: sly.DatasetInfo,
    dst_dataset: sly.DatasetInfo,
    dst_project_meta: sly.ProjectMeta,
    detectors: List[Callable],
    progress,
):
    for batch in g.Api.video.get_list_generator(src_dataset.id, batch_size=1):
        for video in batch:
            timer = {}
            dst_name = video.name
            dst_video_meta = {
                **video.meta,
                "anonymized_objects": True,
                "src_video_id": video.id,
            }
            video_path = os.path.join(g.APP_DATA_DIR, "videos", video.name)

            t = time.time()
            g.Api.video.download_path(
                video.id,
                video_path,
            )
            timer["download"] = round(time.time() - t, 3)

            t = time.time()

            # the original is kept untouched: its data streams are copied back in below
            src_video_path = video_path
            if g.STATE.resize_videos:
                video_path = resize_video(src_video_path, g.STATE.resize_percentage)
            # OpenCV gives up after OPENCV_FFMPEG_READ_ATTEMPTS non-video packets in a row, and a
            # GoPro file interleaves audio, timecode and gpmd telemetry between its frames: on a 4K
            # HERO5 clip it stopped after 14 of 15960 frames. Decode a video-only copy instead; the
            # original still supplies the streams restored below.
            read_path = video_only_copy(video_path)
            cap = cv2.VideoCapture(read_path)
            # not rounded to an int: truncating 29.97 to 29 stretches the output and would
            # desynchronize the restored streams from the video
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                sly.logger.warning(
                    f"Could not read fps of video (id: {video.id}), falling back to 25"
                )
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video_path = os.path.join(g.APP_DATA_DIR, "videos", f"anonymized_{video.name}")
            out = cv2.VideoWriter(
                out_video_path,
                fourcc,
                fps,
                (width, height),
            )
            timer["open video"] = round(time.time() - t, 3)
            t = time.time()
            expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or video.frames_count
            written_frames = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                dets = []
                for _, detector_dets, _ in run_detectors(detectors, frame):
                    dets.extend(detector_dets)
                frame = obfuscate_objects(
                    frame,
                    [d[:4] for d in dets],
                    g.STATE.obfuscate_shape,
                    g.STATE.obfuscate_method,
                )
                out.write(frame)
                written_frames += 1
                progress.update(1)
            cap.release()
            out.release()
            if read_path != video_path:
                sly.fs.silent_remove(read_path)
            if expected_frames and written_frames < expected_frames * 0.99:
                raise RuntimeError(
                    f"Video {video.name} (id: {video.id}) stopped decoding after {written_frames} "
                    f"of {expected_frames} frames. Nothing was uploaded for it."
                )
            timer["process video"] = round(time.time() - t, 3)

            t = time.time()
            fix_codec(out_video_path)  # fix codec for Video Labeling tool
            restore_source_streams(out_video_path, src_video_path)
            if video_path != src_video_path:
                sly.fs.silent_remove(video_path)

            dst_video_info = g.Api.video.upload_path(
                dst_dataset.id, dst_name, out_video_path, dst_video_meta
            )
            key_id_map = sly.KeyIdMap()
            src_ann = sly.VideoAnnotation.from_json(
                g.Api.video.annotation.download(video.id), dst_project_meta, key_id_map
            )
            g.Api.video.annotation.append(dst_video_info.id, src_ann, key_id_map)
            timer["upload"] = round(time.time() - t, 3)
            timer["start to finish"] = round(sum(timer.values()), 1)
            sly.logger.debug(f"Finished processing video (id: {video.id})", extra=timer)


def get_total_items(ds_tree: Dict[sly.DatasetInfo, Dict], func: Callable):
    count = 0
    for ds_info, children in ds_tree.items():
        count += func(ds_info)
        if children:
            count += get_total_items(children, func)
    return count


def get_selected_ds(ds_tree, id: int) -> List[str]:
    for ds_info, children in ds_tree.items():
        if ds_info.id == id:
            return ds_info
        if children:
            result = get_selected_ds(children, id)
            if result is not None:
                return result


def download_models(dir: str = None):
    download_yunet_model(Path(dir, "face_detection_yunet_2023mar.onnx"))
    download_egoblur_model(Path(dir, "ego_blur_lp.jit"))


def get_detectors():
    if g.STATE.target == g.Model.YUNET:
        return [detect_faces_yunet]
    elif g.STATE.target == g.Model.EGOBLUR:
        return [detect_lp_egoblur]
    else:
        return [detect_faces_yunet, detect_lp_egoblur]


def main():
    def create_ds_recursive(ds_tree, dst_project, parent_id=None):
        for ds_info, children in ds_tree.items():
            current_ds = create_dst_dataset(ds_info, dst_project, parent_id)
            run_func(ds_info, current_ds, dst_project_meta, detectors, progress)
            if children:
                create_ds_recursive(children, dst_project, current_ds.id)

    src_project = g.Api.project.get_info_by_id(g.STATE.selected_project)
    dst_project = create_dst_project(src_project)
    dst_project_meta = sly.ProjectMeta.from_json(g.Api.project.get_meta(dst_project.id))

    src_ds_tree = g.Api.dataset.get_tree(src_project.id)
    if g.STATE.selected_dataset:
        selected_dsinfo = get_selected_ds(src_ds_tree, g.STATE.selected_dataset)
        src_ds_tree = {selected_dsinfo: None}

    if src_project.type == str(sly.ProjectType.IMAGES):
        run_func = run_images
        total_cb = lambda x: x.items_count
    else:
        run_func = run_videos
        total_cb = lambda x: sum([video.frames_count for video in g.Api.video.get_list(x.id)])
    total = get_total_items(src_ds_tree, total_cb)

    with sly.tqdm.tqdm(total=total) as progress:
        detectors = get_detectors()
        create_ds_recursive(src_ds_tree, dst_project)
