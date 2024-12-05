# dds cloudapi for Grounding DINO 1.5
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

import os
import cv2
import json
import torch
import tempfile
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

"""
Hyper parameters
"""
API_TOKEN = "Your API token"
TEXT_PROMPT = "car . building ."
IMG_PATH = "notebooks/images/cars.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BOX_THRESHOLD = 0.2
WITH_SLICE_INFERENCE = False
SLICE_WH = (480, 480)
OVERLAP_RATIO = (0.2, 0.2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_dinox_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Prompt DINO-X with Text for Box Prompt Generation with Cloud API
"""
# Step 1: initialize the config
token = API_TOKEN
config = Config(token)

# Step 2: initialize the client
client = Client(config)

# Step 3: run the task by DetectionTask class
# image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
# if you are processing local image file, upload them to DDS server to get the image url

classes = [x.strip().lower() for x in TEXT_PROMPT.split('.') if x]
class_name_to_id = {name: id for id, name in enumerate(classes)}
class_id_to_name = {id: name for name, id in class_name_to_id.items()}

if WITH_SLICE_INFERENCE:
    def callback(image_slice: np.ndarray) -> sv.Detections:
        print("Inference on image slice")
        # save the img as temp img file for GD-1.5 API usage
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
            temp_filename = tmpfile.name
        cv2.imwrite(temp_filename, image_slice)
        image_url = client.upload_file(temp_filename)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=TEXT_PROMPT)],
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox],
        )
        client.run_task(task)
        result = task.result
        # detele the tempfile
        os.remove(temp_filename)
        
        input_boxes = []
        confidences = []
        class_ids = []
        objects = result.objects
        for idx, obj in enumerate(objects):
            input_boxes.append(obj.bbox)
            confidences.append(obj.score)
            cls_name = obj.category.lower().strip()
            class_ids.append(class_name_to_id[cls_name])
        # ensure input_boxes with shape (_, 4)
        input_boxes = np.array(input_boxes).reshape(-1, 4)
        class_ids = np.array(class_ids)
        confidences = np.array(confidences)
        return sv.Detections(xyxy=input_boxes, confidence=confidences, class_id=class_ids)
    
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=SLICE_WH,
        overlap_ratio_wh=OVERLAP_RATIO,
        iou_threshold=0.5,
        overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION
        )
    detections = slicer(cv2.imread(IMG_PATH))
    class_names = [class_id_to_name[id] for id in detections.class_id]
    confidences = detections.confidence
    class_ids = detections.class_id
    input_boxes = detections.xyxy
else:
    image_url = client.upload_file(IMG_PATH)

    task = DinoxTask(
        image_url=image_url,
        prompts=[TextPrompt(text=TEXT_PROMPT)],
        bbox_threshold=0.25,
        targets=[DetectionTarget.BBox],
    )

    client.run_task(task)
    result = task.result

    objects = result.objects  # the list of detected objects


    input_boxes = []
    confidences = []
    class_names = []
    class_ids = []

    for idx, obj in enumerate(objects):
        input_boxes.append(obj.bbox)
        confidences.append(obj.score)
        cls_name = obj.category.lower().strip()
        class_names.append(cls_name)
        class_ids.append(class_name_to_id[cls_name])

    input_boxes = np.array(input_boxes)
    class_ids = np.array(class_ids)

"""
Init SAM 2 Model and Predict Mask with Box Prompt
"""

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

image = Image.open(IMG_PATH)

sam2_predictor.set_image(np.array(image.convert("RGB")))

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)


"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)


"""
Visualization the Predict Results
"""

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

"""
Visualize image with supervision useful API
"""
img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "dinox_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "dinox_sam2_annotated_image_with_mask.jpg"), annotated_frame)

print(f'Annotated image has already been saved as to "{OUTPUT_DIR}"')

"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    print("Start dumping the annotation...")
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # FIXME: class_names should be a list of strings without spaces
    class_names = [class_name.strip() for class_name in class_names]
    # save the results in standard format
    results = {
        "image_path": IMG_PATH,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_dinox_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f'Annotation has already been saved to "{OUTPUT_DIR}"')