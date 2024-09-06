# dds cloudapi for Grounding DINO 1.5
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget

import os
import cv2
import json
import torch
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
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"
GROUNDING_MODEL = DetectionModel.GDino1_5_Pro # DetectionModel.GDino1_6_Pro
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/grounded_sam2_gd1.5_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
Prompt Grounding DINO 1.5 with Text for Box Prompt Generation with Cloud API
"""
# Step 1: initialize the config
token = API_TOKEN
config = Config(token)

# Step 2: initialize the client
client = Client(config)

# Step 3: run the task by DetectionTask class
# image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
# if you are processing local image file, upload them to DDS server to get the image url
img_path = IMG_PATH
image_url = client.upload_file(img_path)

task = DetectionTask(
    image_url=image_url,
    prompts=[TextPrompt(text=TEXT_PROMPT)],
    targets=[DetectionTarget.BBox],  # detect bbox
    model=GROUNDING_MODEL,  # detect with GroundingDino-1.5-Pro model
)

client.run_task(task)
result = task.result

objects = result.objects  # the list of detected objects


input_boxes = []
confidences = []
class_names = []

for idx, obj in enumerate(objects):
    input_boxes.append(obj.bbox)
    confidences.append(obj.score)
    class_names.append(obj.category)

input_boxes = np.array(input_boxes)

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

image = Image.open(img_path)

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

class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]

"""
Visualize image with supervision useful API
"""
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator()
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # FIXME: class_names should be a list of strings without spaces
    class_names = [class_name.strip() for class_name in class_names]
    # save the results in standard format
    results = {
        "image_path": img_path,
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
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_gd1.5_image_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)
