import logging
import os
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np
from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from shapely.geometry import Polygon, box

from detection import Detection

labels = read_label_file("labels.txt")
interpreter = make_interpreter(
    "ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite"
)
interpreter.allocate_tensors()


class ReolinkVideo:
    REOLINK_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
    FRIENDLY_TIMESTAMP_FORMAT = "%Y-%m-%d %H-%M-%S"
    VALID_DETECTION_LABELS = ["person"]

    def __init__(
        self,
        path: str,
        filename: str,
        extension: str,
    ):
        self.path = path
        self.filename = filename
        self.extension = extension

        (
            self.camera_name,
            self.camera_num,
            self.timestamp,
        ) = self.split_reolink_filename(filename)

    @classmethod
    def split_reolink_filename(
        cls, filename: str
    ) -> Tuple[str, str, datetime]:
        # Example: "Front Door_01_20210511082721"
        filename_parts = filename.rsplit("_", 2)
        camera_name = filename_parts[0]
        camera_num = filename_parts[1]
        timestamp = datetime.strptime(
            filename_parts[2], cls.REOLINK_TIMESTAMP_FORMAT
        )
        return camera_name, camera_num, timestamp

    @property
    def friendly_timestamp(self) -> str:
        return self.timestamp.strftime(self.FRIENDLY_TIMESTAMP_FORMAT)

    @property
    def filename_with_ext(self) -> str:
        return f"{self.filename}.{self.extension}"

    @property
    def full_path(self) -> str:
        return os.path.join(self.path, self.filename_with_ext)

    def friendly_filename(self, extension: str = None) -> str:
        if extension is None:
            extension = self.extension
        return f"{self.friendly_timestamp} ({self.camera_name}).{extension}"

    def _detection_in_roi(self, detection: Detection, roi: Polygon) -> bool:
        object = box(
            detection.x_min,
            detection.y_min,
            detection.x_max,
            detection.y_max,
        )
        return object.intersects(roi)

    def _is_accepted_detection(
        self, detection: Detection, roi: Polygon = None
    ) -> bool:
        if detection.label not in self.VALID_DETECTION_LABELS:
            return False
        if roi is None:
            return True
        else:
            return self._detection_in_roi(detection, roi)

    def _get_detections(
        self, frame: np.ndarray, min_confidence: float
    ) -> List[Detection]:
        _, scale = common.set_resized_input(
            interpreter,
            (frame.shape[1], frame.shape[0]),
            lambda size: cv2.resize(frame, size),
        )
        interpreter.invoke()
        objs = detect.get_objects(interpreter, min_confidence, scale)

        return [
            Detection(
                labels.get(obj.id),
                obj.score,
                obj.bbox.xmax,
                obj.bbox.xmin,
                obj.bbox.ymax,
                obj.bbox.ymin,
            )
            for obj in objs
        ]

    def _scan_frame(
        self, frame: np.ndarray, min_confidence: float, roi: Polygon
    ) -> Tuple[bool, Detection]:
        detections = self._get_detections(frame, min_confidence=min_confidence)
        for detection in detections:
            if self._is_accepted_detection(detection, roi):
                logging.info(
                    f"ACCEPTED {self.filename_with_ext}, "
                    f"{detection.label} detected "
                    f"({100 * detection.confidence}%)"
                )
                return True, detection
        return False, None

    def is_accepted(
        self,
        min_confidence: float = 0.5,
        roi: Polygon = None,
    ) -> Tuple[bool, np.ndarray, Detection]:
        logging.info(f"ANALYSING {self.filename_with_ext}")
        video = cv2.VideoCapture(
            os.path.join(self.path, self.filename_with_ext)
        )
        current_frame = 0
        skip_frames = 15  # Roughly 0.5s
        read_ok, frame = video.read()
        if not read_ok:
            logging.error(f"Unable to read {self.filename_with_ext}")

        while read_ok is True:
            if current_frame % skip_frames == 0:
                accepted, detection = self._scan_frame(
                    frame, min_confidence, roi
                )
                if accepted:
                    video.release()
                    return True, frame, detection
            read_ok, frame = video.read()
            current_frame += 1

        logging.info(f"REJECTED {self.filename_with_ext}")
        video.release()
        return False, None, None

    def move(self, target_path: str) -> None:
        target_full_path = os.path.join(target_path, self.friendly_filename())
        os.rename(self.full_path, target_full_path)

    @classmethod
    def _draw_shape(
        cls,
        frame: np.ndarray,
        shape: Polygon,
        colour: Tuple[int, int, int] = (0, 0, 255),
        label: str = None,
    ) -> np.ndarray:
        coords = np.array(shape.exterior.coords, np.int32)
        thickness = 2
        if label:
            frame = cv2.putText(
                frame,
                label,
                (
                    int(shape.bounds[0]) - thickness,
                    int(shape.bounds[1]) - (thickness * 5),
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(colour),
                thickness=thickness,
            )
        return cv2.polylines(frame, [coords], True, colour, thickness)

    @classmethod
    def save_images_from_frame(
        cls,
        frame: np.ndarray,
        detection: Detection,
        outputs: List[str],
        draw_roi: bool = False,
        roi: Polygon = None,
    ) -> None:
        frame = cls._draw_shape(
            frame,
            detection.box,
            label=f"{detection.label.capitalize()} ({detection.confidence}%)",
        )
        if draw_roi is True and roi is not None:
            frame = cls._draw_shape(frame, roi, (255, 0, 0))
        for output in outputs:
            cv2.imwrite(output, frame)
