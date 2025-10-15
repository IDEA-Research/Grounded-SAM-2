import os
import cv2
import json
import torch
import numpy as np
import argparse
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

def parse_args():
    parser = argparse.ArgumentParser(description="Grounded SAM2 Local Demo")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for object detection (e.g., 'car. tire.')")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--sam2_checkpoint", type=str, required=True, help="Path to the SAM2 checkpoint file")
    parser.add_argument("--sam2_config_root", type=str, default=None, help="Optional root path where config files are located")
    parser.add_argument("--sam2_model_config", type=str, required=True, help="Path to the SAM2 model config file")
    parser.add_argument("--grounding_dino_config", type=str, required=True, help="Path to the Grounding DINO config file")
    parser.add_argument("--grounding_dino_checkpoint", type=str, required=True, help="Path to the Grounding DINO checkpoint file")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="Box threshold for Grounding DINO")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for Grounding DINO")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files")
    parser.add_argument("--dump_json_results", action="store_true", help="Dump results in JSON format")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    return parser.parse_args()

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build SAM2 image predictor
    sam2_model = build_sam2(
    config_file=args.sam2_model_config,
    ckpt_path=args.sam2_checkpoint,
    device=args.device,
    config_root=args.sam2_config_root  # Optional
)
    
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Build Grounding DINO model
    grounding_model = load_model(
        model_config_path=args.grounding_dino_config,
        model_checkpoint_path=args.grounding_dino_checkpoint,
        device=args.device
    )

    # Load image and set up text prompt
    text = args.text_prompt
    image_source, image = load_image(args.img_path)
    sam2_predictor.set_image(image_source)

    # Predict boxes, confidences, and labels
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # Process the box prompt for SAM2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # Enable bfloat16 for inference
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Predict masks using SAM2
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Post-process masks
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]

    # Visualize image with supervision API
    img = cv2.imread(args.img_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)


    # Generate unique filenames based on the input image name
    base_name = Path(args.img_path).stem
    annotated_image_path = output_dir / f"{base_name}_groundingdino_annotated.jpg"
    mask_annotated_image_path = output_dir / f"{base_name}_grounded_sam2_mask.jpg"
    json_results_path = output_dir / f"{base_name}_results.json"

    cv2.imwrite(annotated_image_path, annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    
    cv2.imwrite(mask_annotated_image_path, annotated_frame)

    # Dump results in JSON format
    if args.dump_json_results:
        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        input_boxes = input_boxes.tolist()
        scores = scores.tolist()

        results = {
            "image_path": args.img_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        with open(json_results_path, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()