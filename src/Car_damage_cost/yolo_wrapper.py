"""Thin wrapper around an Ultralytics YOLO-seg model.

Returns full-resolution boolean masks built from the polygons (`masks.xy`),
which are already in original-image pixel coordinates. This avoids the
letterbox-padding problem you get from resizing `masks.data` directly.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from schemas import Detection


class YoloSegWrapper:
    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
        class_conf: dict[str, float] | None = None,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.class_conf = class_conf or {}

    @property
    def class_names(self) -> list[str]:
        return [self.model.names[i] for i in sorted(self.model.names)]

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        # run at the lowest threshold we need; filter per-class afterwards
        low = min([self.conf, *self.class_conf.values()])
        res = self.model.predict(
            image_bgr,
            conf=low,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        dets: list[Detection] = []
        if res.masks is None or res.boxes is None:
            return dets

        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i])
            name = self.model.names[cls_id]
            conf = float(res.boxes.conf[i])
            if conf < self.class_conf.get(name, self.conf):
                continue

            poly = res.masks.xy[i]
            if poly is None or len(poly) < 3:
                continue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
            mask = mask.astype(bool)
            area = int(mask.sum())
            if area == 0:
                continue

            x1, y1, x2, y2 = (float(v) for v in res.boxes.xyxy[i].tolist())
            dets.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    mask=mask,
                    area_px=area,
                )
            )
        return dets