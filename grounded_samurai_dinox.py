# libraries for SAMURAI
import os
import cv2
import torch
import numpy as np
import supervision as sv
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

# dds cloudapi for DINO-X
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

"""
Hyperparam for Ground and Tracking
"""
VIDEO_PATH = "demo.mp4"
TEXT_PROMPT = "person."
OUTPUT_VIDEO_PATH = "./tracking_demo.mp4"
SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames"
SAVE_TRACKING_RESULTS_DIR = "./tracking_results"
API_TOKEN_FOR_DINOX = "Your API token"
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
BOX_THRESHOLD = 0.2

"""
Step 1: Environment settings and model initialization for SAM 2
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "/comp_robot/rentianhe/code/samurai/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/samurai/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# video_dir = "notebooks/videos/bedroom"

"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, 
    overwrite=True, 
    image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt DINO-X with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)

# Step 1: initialize the config
config = Config(API_TOKEN_FOR_DINOX)

# Step 2: initialize the client
client = Client(config)

# Step 3: run the task by DetectionTask class
# image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
# if you are processing local image file, upload them to DDS server to get the image url
image_url = client.upload_file(img_path)

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

for idx, obj in enumerate(objects):
    input_boxes.append(obj.bbox)
    confidences.append(obj.score)
    class_names.append(obj.category)

input_boxes = np.array(input_boxes)

print(input_boxes)

# process the detection results
OBJECTS = class_names

print(OBJECTS)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

# Using box prompt
if PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
        break

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
    os.makedirs(SAVE_TRACKING_RESULTS_DIR)

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

for frame_idx, segments in video_segments.items():
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    
    object_ids = list(segments.keys())
    masks = list(segments.values())
    masks = np.concatenate(masks, axis=0)
    
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
        mask=masks, # (n, h, w)
        class_id=np.array(object_ids, dtype=np.int32),
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


"""
Step 6: Convert the annotated frames to video
"""

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")


create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)