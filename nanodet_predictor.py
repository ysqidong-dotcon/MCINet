import os
import cv2
import numpy as np
import torch

from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.model.arch import build_model
from nanodet.util import Logger, load_model_weight
from nanodet.util.config import cfg, load_config


def warp_boxes(boxes, M, width, height):
    if boxes.shape[0] == 0:
        return boxes
    num_boxes = boxes.shape[0]
    corners = np.zeros((num_boxes * 4, 3), dtype=np.float32)
    corners[0::4, :2] = boxes[:, [0, 1]]  # top-left
    corners[1::4, :2] = boxes[:, [2, 1]]  # top-right
    corners[2::4, :2] = boxes[:, [2, 3]]  # bottom-right
    corners[3::4, :2] = boxes[:, [0, 3]]  # bottom-left
    corners[:, 2] = 1.0
    warped = corners @ M.T
    warped = warped.reshape(-1, 4, 2)
    min_xy = np.min(warped, axis=1)
    max_xy = np.max(warped, axis=1)
    new_boxes = np.hstack((min_xy, max_xy))
    new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, width)
    new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, height)
    return new_boxes


class NanoDetPredictor:
    def __init__(self, config_path: str, model_path: str, score_thresh: float = 0.35, device: str = "cuda:0"):
        load_config(cfg, config_path)
        self.cfg = cfg
        self.device = device
        self.score_thresh = score_thresh
        self.class_names = self.cfg.class_names

        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location='cpu')
        logger = Logger(local_rank=0)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(self.device).eval()

        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def preprocess(self, image_path: str):
        img = cv2.imread(image_path)
        meta = {
            'img_info': {
                'file_name': os.path.basename(image_path),
                'height': img.shape[0],
                'width': img.shape[1],
                'id': 0,
            },
            'raw_img': img,
            'img': img
        }
        dst_shape = img.shape[:2]
        data = self.pipeline(None, meta, dst_shape=dst_shape)


        data["img"] = torch.from_numpy(data["img"]).permute(2, 0, 1).float()

        return data

    def predict(self, image_path: str):
        data = self.preprocess(image_path)

        img_tensor = data["img"]
        batch_img = stack_batch_img([img_tensor], divisible=32).to(self.device)


        with torch.no_grad():
            pred = self.model(batch_img)
            raw_result = self.model.head.post_process(pred, data)

        det_result = raw_result[0]
        bboxes, labels, scores = [], [], []

        for class_id, dets in det_result.items():
            for det in dets:
                bboxes.append(det[:4])
                scores.append(det[4])
                labels.append(class_id)


        bboxes = np.array(bboxes)
        scores = np.array(scores)
        labels = np.array(labels)

        return data["raw_img"], {
            "bboxes": bboxes,
            "labels": labels,
            "scores": scores
        }

    def draw(self, img, bboxes, labels, scores):
        for bbox, label, score in zip(bboxes, labels, scores):
            if score < self.score_thresh:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{self.class_names[label]}: {score:.2f}"
            cv2.putText(img, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img