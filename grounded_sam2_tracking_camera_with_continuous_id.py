import copy
import os

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

# Setup environment
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class GroundingDinoPredictor:
    """
    Wrapper for using a GroundingDINO model for zero-shot object detection.
    """

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        """
        Initialize the GroundingDINO predictor.
        Args:
            model_id (str): HuggingFace model ID to load.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
            device
        )

    def predict(
        self,
        image: "PIL.Image.Image",
        text_prompts: str,
        box_threshold=0.25,
        text_threshold=0.25,
    ):
        """
        Perform object detection using text prompts.
        Args:
            image (PIL.Image.Image): Input RGB image.
            text_prompts (str): Text prompt describing target objects.
            box_threshold (float): Confidence threshold for box selection.
            text_threshold (float): Confidence threshold for text match.
        Returns:
            Tuple[Tensor, List[str]]: Bounding boxes and matched class labels.
        """
        inputs = self.processor(
            images=image, text=text_prompts, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        return results[0]["boxes"], results[0]["labels"]


class SAM2ImageSegmentor:
    """
    Wrapper class for SAM2-based segmentation given bounding boxes.
    """

    def __init__(self, sam_model_cfg: str, sam_model_ckpt: str, device="cuda"):
        """
        Initialize the SAM2 image segmentor.
        Args:
            sam_model_cfg (str): Path to the SAM2 config file.
            sam_model_ckpt (str): Path to the SAM2 checkpoint file.
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        sam_model = build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = SAM2ImagePredictor(sam_model)

    def set_image(self, image: np.ndarray):
        """
        Set the input image for segmentation.
        Args:
            image (np.ndarray): RGB image array with shape (H, W, 3).
        """
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor):
        """
        Predict segmentation masks from given bounding boxes.
        Args:
            boxes (torch.Tensor): Bounding boxes as (N, 4) tensor.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - masks: Binary masks per box, shape (N, H, W)
                - scores: Confidence scores for each mask
                - logits: Raw logits from the model
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        # Normalize shape to (N, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits


class IncrementalObjectTracker:
    def __init__(
        self,
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text="car.",
        detection_interval=20,
    ):
        """
        Initialize an incremental object tracker using GroundingDINO and SAM2.
        Args:
            grounding_model_id (str): HuggingFace model ID for GroundingDINO.
            sam2_model_cfg (str): Path to SAM2 model config file.
            sam2_ckpt_path (str): Path to SAM2 model checkpoint.
            device (str): Device to run the models on ('cuda' or 'cpu').
            prompt_text (str): Initial text prompt for detection.
            detection_interval (int): Frame interval between full detections.
        """
        self.device = device
        self.detection_interval = detection_interval
        self.prompt_text = prompt_text

        # Load models
        self.grounding_predictor = GroundingDinoPredictor(
            model_id=grounding_model_id, device=device
        )
        self.sam2_segmentor = SAM2ImageSegmentor(
            sam_model_cfg=sam2_model_cfg,
            sam_model_ckpt=sam2_ckpt_path,
            device=device,
        )
        self.video_predictor = build_sam2_video_predictor(
            sam2_model_cfg, sam2_ckpt_path
        )

        # Initialize inference state
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.total_frames = 0
        self.objects_count = 0
        self.frame_cache_limit = detection_interval - 1  # or higher depending on memory

        # Store tracking results
        self.last_mask_dict = MaskDictionaryModel()
        self.track_dict = MaskDictionaryModel()

    def add_image(self, image_np: np.ndarray):
        """
        Add a new image frame to the tracker and perform detection or tracking update.
        Args:
            image_np (np.ndarray): Input RGB image as (H, W, 3), dtype=uint8.
        Returns:
            np.ndarray: Annotated image with object masks and labels.
        """
        import numpy as np
        from PIL import Image

        img_pil = Image.fromarray(image_np)

        # Step 1: Perform detection every detection_interval frames
        if self.total_frames % self.detection_interval == 0:
            if (
                self.inference_state["video_height"] is None
                or self.inference_state["video_width"] is None
            ):
                (
                    self.inference_state["video_height"],
                    self.inference_state["video_width"],
                ) = image_np.shape[:2]

            if self.inference_state["images"].shape[0] > self.frame_cache_limit:
                print(
                    f"[Reset] Resetting inference state after {self.frame_cache_limit} frames to free memory."
                )
                self.inference_state = self.video_predictor.init_state()
                self.inference_state["images"] = torch.empty(
                    (0, 3, 1024, 1024), device=self.device
                )
                (
                    self.inference_state["video_height"],
                    self.inference_state["video_width"],
                ) = image_np.shape[:2]

            # 1.1 GroundingDINO object detection
            boxes, labels = self.grounding_predictor.predict(img_pil, self.prompt_text)
            if boxes.shape[0] == 0:
                return

            # 1.2 SAM2 segmentation from detection boxes
            self.sam2_segmentor.set_image(image_np)
            masks, scores, logits = self.sam2_segmentor.predict_masks_from_boxes(boxes)

            # 1.3 Build MaskDictionaryModel
            mask_dict = MaskDictionaryModel(
                promote_type="mask", mask_name=f"mask_{self.total_frames:05d}.npy"
            )
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(self.device),
                box_list=torch.tensor(boxes),
                label_list=labels,
            )

            # 1.4 Object ID tracking and IOU-based update
            self.objects_count = mask_dict.update_masks(
                tracking_annotation_dict=self.last_mask_dict,
                iou_threshold=0.3,
                objects_count=self.objects_count,
            )

            # 1.5 Reset video tracker state
            frame_idx = self.video_predictor.add_new_frame(
                self.inference_state, image_np
            )
            self.video_predictor.reset_state(self.inference_state)

            for object_id, object_info in mask_dict.labels.items():
                frame_idx, _, _ = self.video_predictor.add_new_mask(
                    self.inference_state,
                    frame_idx,
                    object_id,
                    object_info.mask,
                )

            self.track_dict = copy.deepcopy(mask_dict)
            self.last_mask_dict = mask_dict

        else:
            # Step 2: Use incremental tracking for intermediate frames
            frame_idx = self.video_predictor.add_new_frame(
                self.inference_state, image_np
            )

        # Step 3: Tracking propagation using the video predictor
        frame_idx, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
        )

        # Step 4: Update the mask dictionary based on tracked masks
        frame_masks = MaskDictionaryModel()
        for i, obj_id in enumerate(obj_ids):
            out_mask = video_res_masks[i] > 0.0
            object_info = ObjectInfo(
                instance_id=obj_id,
                mask=out_mask[0],
                class_name=self.track_dict.get_target_class_name(obj_id),
                logit=self.track_dict.get_target_logit(obj_id),
            )
            object_info.update_box()
            frame_masks.labels[obj_id] = object_info
            frame_masks.mask_name = f"mask_{frame_idx:05d}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        self.last_mask_dict = copy.deepcopy(frame_masks)

        # Step 5: Build mask array
        H, W = image_np.shape[:2]
        mask_img = torch.zeros((H, W), dtype=torch.int32)
        for obj_id, obj_info in self.last_mask_dict.labels.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_array = mask_img.cpu().numpy()

        # Step 6: Visualization
        annotated_frame = self.visualize_frame_with_mask_and_metadata(
            image_np=image_np,
            mask_array=mask_array,
            json_metadata=self.last_mask_dict.to_dict(),
        )

        print(f"[Tracker] Total processed frames: {self.total_frames}")
        self.total_frames += 1
        torch.cuda.empty_cache()
        return annotated_frame

    def set_prompt(self, new_prompt: str):
        """
        Dynamically update the GroundingDINO prompt and reset tracking state
        to force a new object detection.
        """
        self.prompt_text = new_prompt
        self.total_frames = 0  # Trigger immediate re-detection
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty(
            (0, 3, 1024, 1024), device=self.device
        )
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        print(f"[Prompt Updated] New prompt: '{new_prompt}'. Tracker state reset.")

    def save_current_state(self, output_dir, raw_image: np.ndarray = None):
        """
        Save the current mask, metadata, raw image, and annotated result.
        Args:
            output_dir (str): The root output directory.
            raw_image (np.ndarray, optional): The original input image (RGB).
        """
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        image_data_dir = os.path.join(output_dir, "images")
        vis_data_dir = os.path.join(output_dir, "result")

        os.makedirs(mask_data_dir, exist_ok=True)
        os.makedirs(json_data_dir, exist_ok=True)
        os.makedirs(image_data_dir, exist_ok=True)
        os.makedirs(vis_data_dir, exist_ok=True)

        frame_masks = self.last_mask_dict

        # Ensure mask_name is valid
        if not frame_masks.mask_name or not frame_masks.mask_name.endswith(".npy"):
            frame_masks.mask_name = f"mask_{self.total_frames:05d}.npy"

        base_name = f"image_{self.total_frames:05d}"

        # Save segmentation mask
        mask_img = torch.zeros(frame_masks.mask_height, frame_masks.mask_width)
        for obj_id, obj_info in frame_masks.labels.items():
            mask_img[obj_info.mask == True] = obj_id
        np.save(
            os.path.join(mask_data_dir, frame_masks.mask_name),
            mask_img.numpy().astype(np.uint16),
        )

        # Save metadata as JSON
        json_path = os.path.join(json_data_dir, base_name + ".json")
        frame_masks.to_json(json_path)

        # Save raw input image
        if raw_image is not None:
            image_bgr = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image_data_dir, base_name + ".jpg"), image_bgr)

            # Save annotated image with mask, bounding boxes, and labels
            annotated_image = self.visualize_frame_with_mask_and_metadata(
                image_np=raw_image,
                mask_array=mask_img.numpy().astype(np.uint16),
                json_metadata=frame_masks.to_dict(),
            )
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(vis_data_dir, base_name + "_annotated.jpg"), annotated_bgr
            )
            print(
                f"[Saved] {base_name}.jpg and {base_name}_annotated.jpg saved successfully."
            )

    def visualize_frame_with_mask_and_metadata(
        self,
        image_np: np.ndarray,
        mask_array: np.ndarray,
        json_metadata: dict,
    ):
        image = image_np.copy()
        H, W = image.shape[:2]

        # Step 1: Parse metadata and build object entries
        metadata_lookup = json_metadata.get("labels", {})

        all_object_ids = []
        all_object_boxes = []
        all_object_classes = []
        all_object_masks = []

        for obj_id_str, obj_info in metadata_lookup.items():
            instance_id = obj_info.get("instance_id")
            if instance_id is None or instance_id == 0:
                continue
            if instance_id not in np.unique(mask_array):
                continue

            object_mask = mask_array == instance_id
            all_object_ids.append(instance_id)
            x1 = obj_info.get("x1", 0)
            y1 = obj_info.get("y1", 0)
            x2 = obj_info.get("x2", 0)
            y2 = obj_info.get("y2", 0)
            all_object_boxes.append([x1, y1, x2, y2])
            all_object_classes.append(obj_info.get("class_name", "unknown"))
            all_object_masks.append(object_mask[None])  # Shape (1, H, W)

        # Step 2: Check if valid objects exist
        if len(all_object_ids) == 0:
            print("No valid object instances found in metadata.")
            return image

        # Step 3: Sort by instance ID
        paired = list(
            zip(all_object_ids, all_object_boxes, all_object_masks, all_object_classes)
        )
        paired.sort(key=lambda x: x[0])

        all_object_ids = [p[0] for p in paired]
        all_object_boxes = [p[1] for p in paired]
        all_object_masks = [p[2] for p in paired]
        all_object_classes = [p[3] for p in paired]

        # Step 4: Build detections
        all_object_masks = np.concatenate(all_object_masks, axis=0)
        detections = sv.Detections(
            xyxy=np.array(all_object_boxes),
            mask=all_object_masks,
            class_id=np.array(all_object_ids, dtype=np.int32),
        )
        labels = [
            f"{instance_id}: {class_name}"
            for instance_id, class_name in zip(all_object_ids, all_object_classes)
        ]

        # Step 5: Annotate image
        annotated_frame = image.copy()
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = mask_annotator.annotate(annotated_frame, detections)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        return annotated_frame


import os

import cv2
import torch
from utils.common_utils import CommonUtils


def main():
    # Parameter settings
    output_dir = "./outputs"
    prompt_text = "hand."
    detection_interval = 20
    max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize the object tracker
    tracker = IncrementalObjectTracker(
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
    )
    tracker.set_prompt("person.")

    # Open the camera (or replace with local video file, e.g., cv2.VideoCapture("video.mp4"))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Cannot open camera.")
        return

    print("[Info] Camera opened. Press 'q' to quit.")
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"[Frame {frame_idx}] Processing live frame...")
            process_image = tracker.add_image(frame_rgb)

            if process_image is None or not isinstance(process_image, np.ndarray):
                print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
                frame_idx += 1
                continue

            # process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Live Inference", process_image_bgr)

            
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("[Info] Quit signal received.")
            #     break

            tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
            frame_idx += 1

            if frame_idx >= max_frames:
                print(f"[Info] Reached max_frames {max_frames}. Stopping.")
                break
    except KeyboardInterrupt:
        print("[Info] Interrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Done] Live inference complete.")


if __name__ == "__main__":
    main()