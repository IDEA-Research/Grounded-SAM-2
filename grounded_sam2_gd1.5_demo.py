# dds cloudapi for Grounding DINO 1.5
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget

import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

"""
Prompt Grounding DINO 1.5 with Text for Box Prompt Generation with Cloud API
"""
# Step 1: initialize the config
token = "3491a2a256fb7ed01b2e757b713c4cb0"
config = Config(token)

# Step 2: initialize the client
client = Client(config)

# Step 3: run the task by DetectionTask class
# image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
# if you are processing local image file, upload them to DDS server to get the image url
img_path = "notebooks/images/cars.jpg"
image_url = client.upload_file(img_path)

task = DetectionTask(
    image_url=image_url,
    prompts=[TextPrompt(text="car")],
    targets=[DetectionTarget.BBox],  # detect bbox
    model=DetectionModel.GDino1_5_Pro,  # detect with GroundingDino-1.5-Pro model
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
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
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
if masks.ndim == 3:
    masks = masks[None]
    scores = scores[None]
    logits = logits[None]
elif masks.ndim == 4:
    masks = masks.squeeze(1)


"""
Visualization the Predict Results
"""

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]
img = cv2.imread(img_path)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks,  # (n, h, w)
)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite("grounded_sam2_annotated_image_with_mask.jpg", annotated_frame)