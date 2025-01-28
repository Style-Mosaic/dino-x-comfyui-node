"""
ComfyUI node for DINO-X API integration
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dds_cloudapi_sdk import Client, Config, TextPrompt
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from PIL import Image


class DinoxDetectorNode:
    """
    A ComfyUI node that uses DINO-X API for object detection and segmentation
    """

    def __init__(self):
        self.output_dir = Path("./outputs/dinox")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "wheel . eye . helmet . mouse . mouth . vehicle . steering wheel . ear . nose"
                    },
                ),
                "api_token": ("STRING", {"default": ""}),
                "bbox_threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")  # Returns both box and mask annotations
    RETURN_NAMES = ("box_annotated", "mask_annotated")
    FUNCTION = "detect_and_annotate"
    CATEGORY = "detection"

    def detect_and_annotate(self, image, text_prompt, api_token, bbox_threshold):
        # Input validation
        if not isinstance(image, (np.ndarray, Image.Image)):
            raise TypeError("Input image must be a PIL Image or numpy array")

        # Convert ComfyUI image (PIL) to OpenCV format
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = image[:, :, ::-1].copy()  # RGB to BGR for OpenCV

        # Input validation
        if not api_token:
            raise ValueError("API token cannot be empty")

        if not text_prompt.strip():
            raise ValueError("Text prompt cannot be empty")

        if bbox_threshold < 0.0 or bbox_threshold > 1.0:
            raise ValueError("bbox_threshold must be between 0.0 and 1.0")

        # Handle alpha channel after type validation
        if image.shape[2] == 4:  # Remove alpha channel if present
            image = image[:, :, :3]

        # Use a context manager for temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                # Save image to temp file
                success = cv2.imwrite(temp_path, image)
                if not success:
                    raise RuntimeError("Failed to save temporary image")
                temp_file.flush()
                os.fsync(temp_file.fileno())

                # Initialize DINO-X API client
                config = Config(api_token)
                client = Client(config)

                # Upload image and run detection
                image_url = client.upload_file(temp_path)
                task = DinoxTask(
                    image_url=image_url,
                    prompts=[TextPrompt(text=text_prompt)],
                    bbox_threshold=bbox_threshold,
                    targets=[DetectionTarget.BBox, DetectionTarget.Mask],
                )
                client.run_task(task)

                # Handle case where no detections are found
                if not task.result or not hasattr(task.result, "objects"):
                    predictions = []
                else:
                    predictions = task.result.objects

                if not predictions:
                    return (
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                    )

                # Process predictions
                classes = [x.strip().lower() for x in text_prompt.split(".") if x]
                class_name_to_id = {name: idx for idx, name in enumerate(classes)}

                boxes = []
                masks = []
                confidences = []
                class_names = []
                class_ids = []

                for obj in predictions:
                    if not hasattr(obj, "mask") or not obj.mask:
                        continue
                    if not hasattr(obj.mask, "counts") or not hasattr(obj.mask, "size"):
                        continue

                    boxes.append(obj.bbox)
                    masks.append(
                        DetectionTask.rle2mask(
                            DetectionTask.string2rle(obj.mask.counts), obj.mask.size
                        )
                    )
                    confidences.append(obj.score)
                    cls_name = obj.category.lower().strip()
                    class_names.append(cls_name)
                    class_ids.append(class_name_to_id[cls_name])

                if not boxes:
                    return (
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                    )

                boxes = np.array(boxes)
                masks = np.array(masks)
                class_ids = np.array(class_ids)
                labels = [
                    f"{class_name} {confidence:.2f}"
                    for class_name, confidence in zip(class_names, confidences)
                ]

                # Create detections object
                detections = sv.Detections(
                    xyxy=boxes,
                    mask=masks.astype(bool),
                    class_id=class_ids,
                )

                # Create annotations
                box_annotated = image.copy()
                box_annotator = sv.BoxAnnotator()
                box_annotated = box_annotator.annotate(
                    scene=box_annotated, detections=detections
                )
                label_annotator = sv.LabelAnnotator()
                box_annotated = label_annotator.annotate(
                    scene=box_annotated, detections=detections, labels=labels
                )
                mask_annotator = sv.MaskAnnotator()
                mask_annotated = mask_annotator.annotate(
                    scene=box_annotated.copy(), detections=detections
                )

                # Convert back to RGB for ComfyUI
                box_annotated = cv2.cvtColor(box_annotated, cv2.COLOR_BGR2RGB)
                mask_annotated = cv2.cvtColor(mask_annotated, cv2.COLOR_BGR2RGB)

                # Convert to PIL images
                box_annotated = Image.fromarray(box_annotated)
                mask_annotated = Image.fromarray(mask_annotated)

                return (box_annotated, mask_annotated)

            finally:
                # Clean up temp file
                try:
                    temp_file.close()
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp file: {str(e)}")


# Register node for ComfyUI
NODE_CLASS_MAPPINGS = {"DinoxDetector": DinoxDetectorNode}

NODE_DISPLAY_NAME_MAPPINGS = {"DinoxDetector": "DINO-X Detector"}
