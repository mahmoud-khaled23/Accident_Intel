"""Thin wrapper around an Ultralytics YOLO-seg model.

Returns full-resolution boolean masks built from the polygons (`masks.xy`),
which are already in original-image pixel coordinates. This avoids the
letterbox-padding problem you get from resizing `masks.data` directly.
"""
from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from .schemas import Detection


class YoloSegWrapper:
    """Convenience wrapper around a YOLO instance-segmentation model.

    Loads the model from a weights file and exposes a single :meth:`predict`
    method that returns :class:`~schemas.Detection` objects with full-resolution
    boolean masks in the original image coordinate space.

    Attributes:
        model: The underlying :class:`~ultralytics.YOLO` model.
        conf: Global confidence threshold; detections below this are dropped.
        iou: IoU threshold for non-maximum suppression.
        imgsz: Inference image size in pixels (longest side).
        device: PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
            ``None`` lets Ultralytics choose.
        class_conf: Per-class confidence overrides that take precedence over
            the global :attr:`conf` threshold.
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        device: str | None = None,
        class_conf: dict[str, float] | None = None,
    ):
        """Load the YOLO model from ``weights``.

        Args:
            weights: Path to the ``.pt`` YOLO weights file.
            conf: Global confidence threshold applied to all classes unless
                overridden by ``class_conf``.
            iou: Non-maximum suppression IoU threshold.
            imgsz: Inference input size (longest side, pixels).
            device: PyTorch device string (``None`` = auto-select).
            class_conf: Optional dict mapping class name → minimum confidence.
                Overrides ``conf`` for those classes.
        """
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.class_conf = class_conf or {}

    @property
    def class_names(self) -> list[str]:
        """Return the model's class names in index order.

        Returns:
            A list of class name strings sorted by class index.
        """
        return [self.model.names[i] for i in sorted(self.model.names)]

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR image and return detections.

        Runs the YOLO model at the lowest confidence threshold needed (to
        honour per-class overrides), then filters the results to the
        per-class or global threshold.  Masks are rasterised from the
        polygon coordinates (``masks.xy``) so they are in original image
        resolution with no letterbox artefacts.

        Detections with fewer than 3 polygon vertices or zero mask area are
        silently discarded.

        Args:
            image_bgr: Input image as a NumPy array in BGR channel order
                with shape ``(H, W, 3)``.

        Returns:
            A list of :class:`~schemas.Detection` objects, one per accepted
            instance, in the order returned by the YOLO model.
        """
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
