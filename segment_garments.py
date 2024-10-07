import os
import json
import dill
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torchvision.ops import box_convert
import numpy as np
from numpy.typing import NDArray
import cv2
import supervision as sv
import pycocotools.mask as mask_util
import coloredlogs

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import grounding_dino.groundingdino.util.inference as gdino_inference


log = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=log)


@dataclass
class GroundedSAMConfig:
    sam2_checkpoint: str
    sam2_model_config: str
    grounding_dino_checkpoint: str
    grounding_dino_config: str
    box_threshold: float
    text_threshold: float


@dataclass
class config:
    base_directory: str = os.path.expanduser("~/atman/SemanticParser")
    input_path: str = "data/output/test_garment_classifier.json"
    output_path: str = "data/output/test_segment_garments"
    grounded_sam_config = GroundedSAMConfig(
        sam2_checkpoint="checkpoints/sam2_hiera_large.pt",
        sam2_model_config="sam2_hiera_l.yaml",
        grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
        grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        box_threshold=0.3,
        text_threshold=0.25
    )


@dataclass
class GroundedSAMPredictions:
    image_path: str
    image_path_base_directory: Optional[str]  # if specified, image_path is relative to this directory
    image_width: int
    image_height: int

    input_boxes: NDArray  # (N, 4)
    masks: NDArray[np.bool_]  # (N, H, W)

    class_names: List[str]  # could be repeated
    anno_ids: NDArray[np.int_]  # (N, ) -> range(N)
    box_confidences: NDArray  # (N, )
    seg_scores: NDArray  # (N, )

    def to_json(self, *, include_segments=True):
        
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle
        
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in self.masks]

        anno_ids = self.anno_ids.tolist()
        input_boxes = self.input_boxes.tolist()
        scores = self.seg_scores.tolist()

        # save the results in standard format
        results = {
            "image_path": self.image_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "anno_id": anno_id,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, anno_id, box, mask_rle, score in zip(self.class_names, anno_ids, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": self.image_width,
            "img_height": self.image_height,
        }

        if not include_segments:
            results["annotations"] = [{k: v for k, v in anno.items() if k != "segmentation"} for anno in results["annotations"]]

        return json.dumps(results)
    
    def visualise_to_image(self, output_path: str):
        confidences = self.box_confidences.numpy().tolist()

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(self.class_names, confidences)
        ]

        image_path = self.image_path if self.image_path_base_directory is None else os.path.join(self.image_path_base_directory, self.image_path)
        img = cv2.imread(image_path)
        detections = sv.Detections(
            xyxy=self.input_boxes,  # (n, 4)
            mask=self.masks,  # (n, h, w)
            class_id=self.anno_ids
        )
        
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        _output_path_parts = os.path.splitext(output_path)
        output_path_original = f"{_output_path_parts[0]}_original{_output_path_parts[1]}"
        output_path_box_only = f"{_output_path_parts[0]}_without_mask{_output_path_parts[1]}"
        output_path_with_mask = f"{_output_path_parts[0]}_with_mask{_output_path_parts[1]}"

        cv2.imwrite(output_path_original, img)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(output_path_box_only, annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(output_path_with_mask, annotated_frame)

    def filter_by_anno_ids(self, anno_ids: List[int]):
        anno_ids_mask = np.isin(self.anno_ids, anno_ids)

        self.input_boxes = self.input_boxes[anno_ids_mask]
        self.masks = self.masks[anno_ids_mask]

        self.class_names = [class_name for class_name, anno_id_mask in zip(self.class_names, anno_ids_mask) if anno_id_mask]
        self.anno_ids = self.anno_ids[anno_ids_mask]
        self.box_confidences = self.box_confidences[anno_ids_mask]
        self.seg_scores = self.seg_scores[anno_ids_mask]


class GroundedSAMPredictor:

    def __init__(self, gsam_config: GroundedSAMConfig, *, image_path_base_directory: Optional[str]=None):
        """
        gsam_config: configuration used to set up Grounded DINO and SAM2
        image_path_base_directory: if specified enables predict to be called with a image path relative to this directory
        """
        self.config = gsam_config
        self.image_path_base_directory = image_path_base_directory
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # build SAM2 image predictor
        self.sam2_model = build_sam2(self.config.sam2_model_config, self.config.sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = gdino_inference.load_model(
            model_config_path=self.config.grounding_dino_config, 
            model_checkpoint_path=self.config.grounding_dino_checkpoint,
            device=device
        )

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    def predict(self, captions: List[str], image_path: str) -> GroundedSAMPredictions:
        # optionally wrangle paths relative to base directory
        image_path_full = image_path
        image_path_base_directory = None
        if self.image_path_base_directory is not None and not os.path.isabs(image_path):
            image_path_base_directory = self.image_path_base_directory
            image_path_full = os.path.join(image_path_base_directory, image_path)
        
        image_source, image = gdino_inference.load_image(image_path_full)

        self.sam2_predictor.set_image(image_source)

        all_boxes = []
        all_confidences = []
        all_labels = []
        for text in captions:
            # returns N predictions for the label 'text'
            #   boxes.shape = (N, 4)
            #   confidences.shape = (N, )
            #   len(labels) = N
            boxes, confidences, labels = gdino_inference.predict(
                model=self.grounding_model,
                image=image,
                caption=text,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
            )
            all_boxes.append(boxes)
            all_confidences.append(confidences)
            all_labels.append(labels)
        
        # now of shape (N*T, 4)
        boxes = torch.cat(all_boxes, dim=0)
        confidences = torch.cat(all_confidences, dim=0)
        labels = [l for sublist in all_labels for l in sublist]

        # process the box prompt for SAM 2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masks, scores, _logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=True,
            )
            # expand dim if only one mask returned
            if scores.ndim == 1:
                masks = masks[np.newaxis, :]
                scores = scores[np.newaxis, :]

        # (N*T, M, H, W)
        sorted_ind = np.argsort(scores, axis=1)

        masks_sorted = np.take_along_axis(masks, sorted_ind[:, :, np.newaxis, np.newaxis], axis=1)
        scores_sorted = np.take_along_axis(scores, sorted_ind, axis=1)

        masks_top = masks_sorted[:, -1, :, :]  # shape (N, M, H, W) -> (N, H, W)
        scores_top = scores_sorted[:, -1]

        anno_ids = np.array(list(range(len(labels))), dtype=int)

        return GroundedSAMPredictions(
            image_path=image_path,
            image_path_base_directory=image_path_base_directory,
            image_width=w,
            image_height=h,
            input_boxes=input_boxes,
            masks=masks_top.astype(bool),
            class_names=labels,
            anno_ids=anno_ids,
            box_confidences=confidences,
            seg_scores=scores_top
        )


if __name__ == "__main__":

    config.input_path = config.input_path if os.path.isabs(config.input_path) else os.path.join(config.base_directory, config.input_path)
    config.output_path = config.output_path if os.path.isabs(config.output_path) else os.path.join(config.base_directory, config.output_path)
    os.makedirs(config.output_path, exist_ok=True)

    with open(config.input_path) as f:
        image_annotations = json.load(f)

    gsam_predictor = GroundedSAMPredictor(config.grounded_sam_config, image_path_base_directory=config.base_directory)

    for idx, (image_path, image_anno) in enumerate(image_annotations.items()):
        for i in range(2):
            log.info(f"Processing {idx}: {image_path}...")

            search_items = [top_category for top_category in image_anno["top_category"]]
            if image_anno["bottom_category"] not in [None, 'not in frame']:
                search_items.append(image_anno["bottom_category"])
            if image_anno["shoes_category"] not in [None, 'not in frame']:
                search_items.append(image_anno["shoes_category"])

            predictions = gsam_predictor.predict(search_items, image_path)
            log.info(f"{predictions.masks.shape[0]} masks found for {len(search_items)} search items")

            output_fname_base = os.path.join(config.output_path, os.path.splitext('--'.join(image_path.split('/')[-2:]))[0])
            output_fname_image = f"{output_fname_base}.jpg"
            output_fname_json = f"{output_fname_base}.json"
            output_fname_dill = f"{output_fname_base}.dill"
            predictions.visualise_to_image(output_fname_image)
            with open(output_fname_json, "w") as f:
                f.write(predictions.to_json())
            with open(output_fname_dill, "wb") as f:
                dill.dump(predictions, f)
