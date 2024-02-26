import os
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
import requests
import supervisely as sly

import globals as g


def updated_project_meta(project_meta: sly.ProjectMeta) -> sly.ProjectMeta:
    """Update project meta to include anonymized faces"""
    obj_classes = project_meta.obj_classes
    if not obj_classes.has_key(g.FACE_CLASS_NAME):
        obj_classes = obj_classes.add(sly.ObjClass(g.FACE_CLASS_NAME, sly.Rectangle))
        project_meta = project_meta.clone(obj_classes=obj_classes)
    tag_metas = project_meta.tag_metas
    if not g.CONFIDENCE_TAG_META_NAME in tag_metas:
        tag_metas = tag_metas.add(
            sly.TagMeta(g.CONFIDENCE_TAG_META_NAME, sly.TagValueType.ANY_NUMBER)
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
        "anonymized_faces": True,
        "src_project_id": src_project.id,
    }
    g.Api.project.update_custom_data(dst_project.id, dst_custom_data)
    return dst_project


def create_dst_dataset(
    src_dataset: sly.DatasetInfo, dst_project: sly.ProjectInfo
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
    )
    return dst_dataset


def download_yunet_model():
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    file_name = "face_detection_yunet_2023mar.onnx"
    model_path = Path(g.APP_DATA_DIR, "models", file_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    with open(model_path, "wb") as f:
        f.write(r.content)
    return model_path.absolute().as_posix()


def get_yunet_model():
    if g.YUNET_MODEl is None:
        backend_id = cv2.dnn.DNN_BACKEND_OPENCV
        target_id = cv2.dnn.DNN_TARGET_CPU
        model_path = download_yunet_model()

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
    model.setInputSize((img.shape[1], img.shape[0]))
    _, faces = model.detect(img)
    if faces is None:
        return []

    def _convert(coord: str):
        return max(0, int(coord))

    res = []
    for face in faces:
        conf = float(face[-1])
        face_coords = [_convert(x) for x in face[:4]]
        res.append([*face_coords, conf])
    return res


def _get_rectangles_mask(size, faces):
    mask = np.zeros(size, dtype=np.uint8)
    for x, y, w, h in faces:
        mask[y : y + h, x : x + w] = 1
    return mask


def blur_faces(img, faces, shape: str):
    for x, y, w, h in faces:
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


def obfuscate_faces(img: np.ndarray, faces: List, shape: str, method: str) -> np.ndarray:
    if shape not in g.AVAILABLE_SHAPES:
        raise ValueError(f"Invalid shape: {shape}")
    if method not in g.AVAILABLE_METHODS:
        raise ValueError(f"Invalid method: {method}")

    if method == g.Method.BLUR:
        return blur_faces(img, faces, shape)
    else:
        if shape == g.Shape.RECTANGLE:
            mask = _get_rectangles_mask(img.shape[:2], faces)
        elif shape == g.Shape.ELLIPSE:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            for x, y, w, h in faces:
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


def update_annotation(dets, annotation: sly.Annotation, project_meta: sly.ProjectMeta):
    face_obj_class = project_meta.get_obj_class(g.FACE_CLASS_NAME)
    conf_tag_meta = project_meta.get_tag_meta(g.CONFIDENCE_TAG_META_NAME)

    labels = []
    for det in dets:
        x, y, w, h, conf = det

        label = sly.Label(
            sly.Rectangle(y, x, y + h, x + w),
            obj_class=face_obj_class,
            tags=[sly.Tag(conf_tag_meta, conf)],
        )
        labels.append(label)
    return annotation.add_labels(labels)


def run_images(
    src_dataset: sly.DatasetInfo,
    dst_dataset: sly.DatasetInfo,
    dst_project_meta: sly.ProjectMeta,
    detector: Callable,
    progress,
):
    for batch in g.Api.image.get_list_generator(src_dataset.id, batch_size=100):
        ann_infos = g.Api.annotation.download_batch(src_dataset.id, [img.id for img in batch])
        anns_dict = {
            ann_info.image_id: sly.Annotation.from_json(
                ann_info.annotation, project_meta=dst_project_meta
            )
            for ann_info in ann_infos
        }

        dst_nps = []
        for image_info in batch:
            img = g.Api.image.download_np(image_info.id)
            dets = detector(img)
            if g.STATE.should_anonymize:
                img = obfuscate_faces(
                    img, [d[:4] for d in dets], g.STATE.obfuscate_shape, g.STATE.obfuscate_method
                )
            dst_nps.append(img)
            if g.STATE.should_save_detections:
                anns_dict[image_info.id] = update_annotation(
                    dets, anns_dict[image_info.id], dst_project_meta
                )

            progress.update(1)

        dst_names = [image_info.name for image_info in batch]
        dst_img_metas = [
            {
                **image_info.meta,
                "anonymized_faces": g.STATE.should_anonymize,
                "src_image_id": image_info.id,
            }
            for image_info in batch
        ]
        dst_images = g.Api.image.upload_nps(
            dataset_id=dst_dataset.id,
            names=dst_names,
            imgs=dst_nps,
            metas=dst_img_metas,
        )
        g.Api.annotation.upload_anns(
            [img.id for img in dst_images], [anns_dict[img.id] for img in batch]
        )


def run_videos(
    src_dataset: sly.DatasetInfo,
    dst_dataset: sly.DatasetInfo,
    dst_project_meta: sly.ProjectMeta,
    detector: Callable,
    progress,
):
    for batch in g.Api.video.get_list_generator(src_dataset.id, batch_size=1):
        for video in batch:
            dst_name = video.name
            dst_video_meta = {**video.meta, "anonymized_faces": True, "src_video_id": video.id}
            video_path = os.path.join(g.APP_DATA_DIR, "videos", video.name)
            g.Api.video.download_path(
                video.id,
                video_path,
            )
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
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
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                dets = detector(frame)
                frame = obfuscate_faces(
                    frame, [d[:4] for d in dets], g.STATE.obfuscate_shape, g.STATE.obfuscate_method
                )
                out.write(frame)
                progress.update(1)
            cap.release()
            out.release()
            dst_video_info = g.Api.video.upload_path(
                dst_dataset.id, dst_name, out_video_path, dst_video_meta
            )
            key_id_map = sly.KeyIdMap()
            src_ann = sly.VideoAnnotation.from_json(
                g.Api.video.annotation.download(video.id), dst_project_meta, key_id_map
            )
            g.Api.video.annotation.append(dst_video_info.id, src_ann, key_id_map)


def get_total_items(datasets: List[sly.DatasetInfo], project_type) -> int:
    if project_type == str(sly.ProjectType.IMAGES):
        return sum([dataset.items_count for dataset in datasets])
    else:
        s = 0
        for dataset in datasets:
            videos = g.Api.video.get_list(dataset.id)
            s += sum([video.frames_count for video in videos])
        return s


def main():
    src_project = g.Api.project.get_info_by_id(g.STATE.selected_project)
    dst_project = create_dst_project(src_project)
    dst_project_meta = sly.ProjectMeta.from_json(g.Api.project.get_meta(dst_project.id))
    if g.STATE.selected_dataset is None:
        datasets = g.Api.dataset.get_list(src_project.id)
    else:
        datasets = [g.Api.dataset.get_info_by_id(g.STATE.selected_dataset)]
    dst_datasets = []
    run_func = run_images if src_project.type == str(sly.ProjectType.IMAGES) else run_videos
    total = get_total_items(datasets, src_project.type)
    with sly.tqdm.tqdm(total=total) as progress:
        detector = detect_faces_yunet

        for dataset in datasets:
            dst_dataset = create_dst_dataset(dataset, dst_project)
            dst_datasets.append(dst_dataset)

            run_func(dataset, dst_dataset, dst_project_meta, detector, progress)
