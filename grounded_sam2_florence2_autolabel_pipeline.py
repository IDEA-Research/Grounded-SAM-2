import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP

"""
Define Some Hyperparam
"""

TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

OUTPUT_DIR = "./outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Init Florence-2 and SAM 2 Model
"""

FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer


"""
We try to support a series of cascaded auto-labelling pipelines with Florence-2 and SAM 2
"""

"""
Auto-Labelling Pipeline 1: Caption/Detailed Caption/More Detailed Caption + Phrase Grounding + Segmentation
"""
def caption_phrase_grounding_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    caption_task_prompt='<CAPTION>',
    output_dir=OUTPUT_DIR
):
    assert caption_task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
    image = Image.open(image_path).convert("RGB")
    
    # image caption
    caption_results = run_florence2(caption_task_prompt, None, florence2_model, florence2_processor, image)
    text_input = caption_results[caption_task_prompt]
    print(f'Image caption for "{image_path}": ', text_input)
    
    # phrase grounding
    grounding_results = run_florence2('<CAPTION_TO_PHRASE_GROUNDING>', text_input, florence2_model, florence2_processor, image)
    grounding_results = grounding_results['<CAPTION_TO_PHRASE_GROUNDING>']
    
    # parse florence-2 detection results
    input_boxes = np.array(grounding_results["bboxes"])
    class_names = grounding_results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_auto_labelling.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_auto_labelling_with_mask.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded SAM 2 Florence-2 Demos", add_help=True)
    parser.add_argument("--image_path", type=str, default="./notebooks/images/cars.jpg", required=True, help="path to image file")
    parser.add_argument("--pipeline", type=str, default="caption_to_phrase_grounding", required=True, help="pipeline to use")
    parser.add_argument("--caption_type", type=str, default="caption", required=False, help="granularity of caption")
    args = parser.parse_args()

    CAPTION_TO_TASK_PROMPT = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>"
    }

    IMAGE_PATH = args.image_path
    PIPELINE = args.pipeline
    CAPTION_TYPE = args.caption_type
    assert CAPTION_TYPE in ["caption", "detailed_caption", "more_detailed_caption"]
    
    print(f"Running pipeline: {PIPELINE} now.")

    if PIPELINE == "caption_to_phrase_grounding":
        # pipeline-1: caption + phrase grounding + segmentation
        caption_phrase_grounding_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            caption_task_prompt=CAPTION_TO_TASK_PROMPT[CAPTION_TYPE],
            image_path=IMAGE_PATH
        )
    else:
        raise NotImplementedError(f"Pipeline: {args.pipeline} is not implemented at this time")
