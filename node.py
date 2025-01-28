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
        """Define the input parameters for the node"""
        return {
            "required": {
                "image": ("IMAGE",),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "wheel . eye . helmet . mouse . mouth . vehicle . steering wheel . ear . nose",
                        "multiline": True,
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
    RETURN_NAMES = ("box_annotated", "binary_mask")
    FUNCTION = "detect_and_annotate"
    CATEGORY = "detection"
    OUTPUT_NODE = True  # Mark as output node since we generate visual output

    @classmethod
    def VALIDATE_INPUTS(cls, api_token, text_prompt, bbox_threshold):
        """Validate input parameters before execution"""
        if not api_token:
            return "API token cannot be empty"

        if not text_prompt.strip():
            return "Text prompt cannot be empty"

        if bbox_threshold < 0.0 or bbox_threshold > 1.0:
            return "bbox_threshold must be between 0.0 and 1.0"

        return True

    @classmethod
    def IS_CHANGED(cls, image, text_prompt, api_token, bbox_threshold):
        """
        Control caching behavior. Since we're making API calls that could
        return different results even with the same inputs, we should
        always run the node.
        """
        return float("NaN")  # Always consider changed due to API nature

    def detect_and_annotate(self, image, text_prompt, api_token, bbox_threshold):
        """
        Process an image with DINO-X API to detect and annotate objects

        Args:
            image: Input image (PIL Image or numpy array)
            text_prompt: Text description of objects to detect
            api_token: DINO-X API token
            bbox_threshold: Detection confidence threshold

        Returns:
            tuple: (box_annotated, mask_annotated) - Two PIL Images with annotations
        """
        try:
            # Handle ComfyUI tensor input
            if isinstance(image, list):  # ComfyUI sends list of tensors
                image = image[0]

            # Convert to numpy array
            if not isinstance(image, np.ndarray):
                try:
                    image = np.array(image)

                    # Handle batch dimension if present (B, H, W, C)
                    if len(image.shape) == 4:
                        if image.shape[0] == 1:  # Single batch
                            image = image[0]  # Remove batch dimension
                        else:
                            raise ValueError(
                                f"Expected batch size 1, got {image.shape[0]}"
                            )
                    elif len(image.shape) != 3:
                        raise ValueError(
                            f"Expected 3 or 4 dimensions, got shape {image.shape}"
                        )
                except:
                    raise ValueError("Failed to convert input to numpy array")

            # Ensure correct dimensions and scale
            if len(image.shape) != 3:
                raise ValueError(
                    f"Expected 3 dimensions (H,W,C), got shape {image.shape}"
                )

            if image.shape[2] not in [3, 4]:
                raise ValueError(f"Expected 3 or 4 channels, got {image.shape[2]}")

            # Scale from [0, 1] to [0, 255] and convert to uint8
            image = (image * 255).astype(np.uint8)

            # Convert to BGR for OpenCV
            image = image[..., :3]  # Take only RGB channels
            image = image[..., ::-1].copy()  # RGB to BGR
        except Exception as e:
            raise ValueError(f"Error processing input image: {str(e)}")

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
                    # Return original image and empty mask
                    empty_mask = np.zeros(
                        (image.shape[0], image.shape[1]), dtype=np.uint8
                    )
                    return (
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                        Image.fromarray(empty_mask, mode="L"),
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
                        Image.fromarray(
                            np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),
                            mode="L",
                        ),
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
                visualization = mask_annotator.annotate(
                    scene=box_annotated.copy(), detections=detections
                )

                # Create binary mask from all detected objects
                binary_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                for mask in masks:
                    binary_mask |= mask.astype(np.uint8) * 255

                # Convert box annotations back to RGB for ComfyUI
                box_annotated = cv2.cvtColor(box_annotated, cv2.COLOR_BGR2RGB)
                box_annotated = Image.fromarray(box_annotated)

                # Convert binary mask to PIL image (mode 'L' for grayscale)
                binary_mask_pil = Image.fromarray(binary_mask, mode="L")

                # Return box annotations and binary mask
                return (box_annotated, binary_mask_pil)

            finally:
                # Clean up temp file
                try:
                    temp_file.close()
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp file: {str(e)}")
