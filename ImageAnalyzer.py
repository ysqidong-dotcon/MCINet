import os
import cv2
import torch
import numpy as np
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from nanodet.data.transform import Pipeline
from nanodet.data.batch_process import stack_batch_img
from nanodet.model.arch import build_model
from nanodet.util import Logger, load_model_weight
from nanodet.util.config import cfg, load_config


class ImageAnalyzer:
    def __init__(self,
                 blip_model_path: str,
                 nanodet_config_path: str,
                 nanodet_model_path: str,
                 device: str = "cuda"):
        self.device = device
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_path)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_path, torch_dtype=torch.float16).to(self.device)
        load_config(cfg, nanodet_config_path)
        self.cfg = cfg
        self.class_names = cfg.class_names

        model = build_model(cfg.model)
        ckpt = torch.load(nanodet_model_path, map_location='cpu')
        logger = Logger(local_rank=0)
        load_model_weight(model, ckpt, logger)
        self.det_model = model.to(self.device).eval()
        self.det_pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def generate_caption(self, img_path: str, prompt: str = "a photo of"):
        raw_image = Image.open(img_path).convert('RGB')
        inputs = self.blip_processor(raw_image, prompt, return_tensors="pt").to(self.device, torch.float16)
        out = self.blip_model.generate(**inputs)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def detect_objects(self, img_path: str, score_thresh: float = 0.35):
        img = cv2.imread(img_path)
        meta = {
            'img_info': {'file_name': os.path.basename(img_path), 'height': img.shape[0], 'width': img.shape[1], 'id': 0},
            'raw_img': img,
            'img': img
        }
        dst_shape = img.shape[:2]
        data = self.det_pipeline(None, meta, dst_shape=dst_shape)
        data["img"] = torch.from_numpy(data["img"]).permute(2, 0, 1).float()
        batch_img = stack_batch_img([data["img"]], divisible=32).to(self.device)

        with torch.no_grad():
            pred = self.det_model(batch_img)
            raw_result = self.det_model.head.post_process(pred, data)[0]

        bboxes, labels, scores = [], [], []
        for class_id, dets in raw_result.items():
            for det in dets:
                if det[4] >= score_thresh:
                    bboxes.append(det[:4])
                    scores.append(det[4])
                    labels.append(class_id)

        return img, {
            "bboxes": np.array(bboxes),
            "labels": np.array(labels),
            "scores": np.array(scores)
        }

    def draw_results(self, img, result):
        bboxes = result["bboxes"]
        labels = result["labels"]
        scores = result["scores"]

        for bbox, label, score in zip(bboxes, labels, scores):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{self.class_names[label]}: {score:.2f}"
            cv2.putText(img, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    def build_prompt(self, caption: str, class_names: list):
        result = ", ".join(class_names)
        prompt = (
            "Please modify the sentence according to the prompts I provided, "
            "ensuring that the prompts are included or emphasized in the sentence.\n"
            f"Prompts: {result}\n"
            f"Sentence: {caption}"
        )
        return prompt

    def query_chatglm3(self, prompt: str, url="http://localhost:8000/ask"):
        try:
            response = requests.post(url, json={"prompt": prompt})
            return response.json().get("response", "[No response]")
        except Exception as e:
            return f"[Error contacting ChatGLM3]: {e}"

    def analyze(self, img_path: str, show: bool = True):
        caption = self.generate_caption(img_path)
        img, result = self.detect_objects(img_path)

        label_ids = np.unique(result["labels"])
        class_names = [self.class_names[label] for label in label_ids]

        print(f"📝 Image Caption: {caption}")
        print("🎯 Detected Classes:")
        for name in class_names:
            print(f"- {name}")

        prompt = self.build_prompt(caption, class_names)
        response = self.query_chatglm3(prompt)
        print("🤖 ChatGLM3 Modified Sentence:")
        print(response)

        # img_vis = self.draw_results(img, result)
        # if show:
        #     cv2.imshow("Result", img_vis)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return caption, class_names, response