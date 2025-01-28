"""
ComfyUI nodes for DINO-X API integration
"""

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from dds_cloudapi_sdk import Client, Config
from dds_cloudapi_sdk.tasks.detection import DetectionTask
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget


class PromptType(str, Enum):
    TEXT = "text"
    UNIVERSAL = "universal"


class DinoxBaseNode:
    """Base class for DINO-X API nodes with common functionality"""

    def __init__(self):
        self.output_dir = Path("./outputs/dinox")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """Convert input tensor to OpenCV format"""
        try:
            if not isinstance(image, torch.Tensor):
                raise ValueError("Expected torch.Tensor input")

            # Ensure BHWC format
            if len(image.shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor [B,H,W,C], got shape {image.shape}"
                )

            # Convert to numpy and scale to [0,255]
            image_np = (image[0].cpu().numpy() * 255).astype(
                np.uint8
            )  # Take first batch

            # Convert to BGR for OpenCV
            return image_np[..., ::-1].copy()  # RGB to BGR
        except Exception as e:
            raise ValueError(f"Error processing input image: {str(e)}")

    def _save_temp_image(self, image: np.ndarray) -> str:
        """Save image to temporary file and return path"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                success = cv2.imwrite(temp_path, image)
                if not success:
                    raise RuntimeError("Failed to save temporary image")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                return temp_path
            except Exception as e:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except:
                    pass
                raise RuntimeError(f"Failed to save temporary image: {str(e)}")


class DinoxDetectorNode(DinoxBaseNode):
    """Node for DINO-X object detection"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C] torch.Tensor
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
                "iou_threshold": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "prompt_type": (["text", "universal"],),
                "return_pose": ("BOOLEAN", {"default": False}),
                "return_hand": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("box_annotated", "binary_mask", "pose_data", "hand_data")
    FUNCTION = "detect_and_annotate"
    CATEGORY = "detection"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(
        cls, api_token, text_prompt, bbox_threshold, iou_threshold, prompt_type
    ):
        if not api_token:
            return "API token cannot be empty"

        if prompt_type == "text" and not text_prompt.strip():
            return "Text prompt cannot be empty when prompt type is 'text'"

        if bbox_threshold < 0.0 or bbox_threshold > 1.0:
            return "bbox_threshold must be between 0.0 and 1.0"

        if iou_threshold < 0.0 or iou_threshold > 1.0:
            return "iou_threshold must be between 0.0 and 1.0"

        return True

    def detect_and_annotate(
        self,
        image: torch.Tensor,
        text_prompt: str,
        api_token: str,
        bbox_threshold: float,
        iou_threshold: float,
        prompt_type: str = "text",
        return_pose: bool = False,
        return_hand: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Process an image with DINO-X API to detect and annotate objects

        Args:
            image: Input image tensor [B,H,W,C]
            text_prompt: Text description of objects to detect
            api_token: DINO-X API token
            bbox_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            prompt_type: Type of prompt ("text" or "universal")
            return_pose: Whether to return pose keypoints
            return_hand: Whether to return hand keypoints

        Returns:
            tuple: (box_annotated, binary_mask, pose_data, hand_data)
        """
        # Convert input tensor to OpenCV format
        image_bgr = self._prepare_image(image)

        # Save to temp file
        temp_path = self._save_temp_image(image_bgr)

        try:
            # Initialize API client
            config = Config(api_token)
            client = Client(config)

            # Prepare targets
            targets = [DetectionTarget.BBox, DetectionTarget.Mask]
            if return_pose:
                targets.append(DetectionTarget.PoseKeypoints)
            if return_hand:
                targets.append(DetectionTarget.HandKeypoints)

            # Prepare prompt
            if prompt_type == "text":
                prompt = {"type": "text", "text": text_prompt}
            else:
                prompt = {"type": "universal", "universal": 1}

            # Upload image and run detection
            image_url = client.upload_file(temp_path)
            task = DinoxTask(
                model="DINO-X-1.0",  # Set model version
                image_url=image_url,
                prompt=prompt,
                bbox_threshold=bbox_threshold,
                iou_threshold=iou_threshold,
                targets=targets,
            )
            client.run_task(task)

            # Handle case where no detections are found
            if not task.result or not hasattr(task.result, "objects"):
                predictions = []
            else:
                predictions = task.result.objects

            # Initialize empty outputs
            h, w = image_bgr.shape[:2]
            empty_mask = torch.zeros((1, h, w), dtype=torch.float32)
            rgb_image = (
                torch.from_numpy(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).float()
                / 255.0
            )
            rgb_image = rgb_image.unsqueeze(0)  # Add batch dimension

            if not predictions:
                return (
                    rgb_image,
                    empty_mask,
                    "[]",  # Empty JSON array for pose data
                    "[]",  # Empty JSON array for hand data
                )

            # Process predictions
            boxes = []
            masks = []
            pose_points = []
            hand_points = []
            confidences = []
            class_names = []

            for obj in predictions:
                if hasattr(obj, "bbox"):
                    boxes.append(obj.bbox)
                    confidences.append(obj.score)
                    class_names.append(obj.category)

                if hasattr(obj, "mask") and obj.mask:
                    if hasattr(obj.mask, "counts") and hasattr(obj.mask, "size"):
                        masks.append(
                            DetectionTask.rle2mask(
                                DetectionTask.string2rle(obj.mask.counts), obj.mask.size
                            )
                        )

                if return_pose and hasattr(obj, "pose"):
                    pose_points.append(obj.pose)

                if return_hand and hasattr(obj, "hand"):
                    hand_points.append(obj.hand)

            if not boxes:
                return (
                    rgb_image,
                    empty_mask,
                    "[]",  # Empty JSON array for pose data
                    "[]",  # Empty JSON array for hand data
                )

            # Create annotations
            boxes = np.array(boxes)
            masks = (
                np.array(masks) if masks else np.zeros((len(boxes), h, w), dtype=bool)
            )
            labels = [
                f"{name} {conf:.2f}" for name, conf in zip(class_names, confidences)
            ]

            # Create detections object
            detections = sv.Detections(
                xyxy=boxes,
                mask=masks.astype(bool),
                class_id=np.arange(len(boxes)),
            )

            # Create box annotations
            box_annotated = image_bgr.copy()
            box_annotator = sv.BoxAnnotator()
            box_annotated = box_annotator.annotate(
                scene=box_annotated, detections=detections
            )
            label_annotator = sv.LabelAnnotator()
            box_annotated = label_annotator.annotate(
                scene=box_annotated, detections=detections, labels=labels
            )

            # Create binary mask
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            for mask in masks:
                binary_mask |= mask.astype(np.uint8)

            # Convert outputs to tensors
            box_annotated = cv2.cvtColor(box_annotated, cv2.COLOR_BGR2RGB)
            box_tensor = torch.from_numpy(box_annotated).float() / 255.0
            box_tensor = box_tensor.unsqueeze(0)  # Add batch dimension

            mask_tensor = torch.from_numpy(binary_mask).float()
            mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

            # Convert keypoints to JSON strings
            pose_json = json.dumps(pose_points) if pose_points else "[]"
            hand_json = json.dumps(hand_points) if hand_points else "[]"

            return (
                box_tensor,
                mask_tensor,
                pose_json,
                hand_json,
            )

        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {str(e)}")


class DinoxRegionVLNode(DinoxBaseNode):
    """Node for DINO-X region-based vision-language tasks"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # [B,H,W,C] torch.Tensor
                "regions": (
                    "STRING",
                    {"multiline": True},
                ),  # JSON array of [x1,y1,x2,y2] coordinates
                "api_token": ("STRING", {"default": ""}),
                "prompt_type": (["text", "universal"],),
                "text_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "return_caption": ("BOOLEAN", {"default": True}),
                "return_roc": ("BOOLEAN", {"default": False}),
                "return_ocr": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("captions", "roc_results", "ocr_results")
    FUNCTION = "analyze_regions"
    CATEGORY = "detection"

    @classmethod
    def VALIDATE_INPUTS(cls, api_token, regions, prompt_type, text_prompt):
        if not api_token:
            return "API token cannot be empty"

        if not regions.strip():
            return "Regions cannot be empty"

        if prompt_type == "text" and not text_prompt.strip():
            return "Text prompt cannot be empty when prompt type is 'text'"

        try:
            import json

            regions_list = json.loads(regions)
            if not isinstance(regions_list, list):
                return "Regions must be a JSON array"
            if len(regions_list) > 1000:
                return "Maximum 1000 regions allowed"
            for region in regions_list:
                if not isinstance(region, list) or len(region) != 4:
                    return "Each region must be an array of 4 coordinates [x1,y1,x2,y2]"
        except Exception as e:
            return f"Invalid regions format: {str(e)}"

        return True

    def analyze_regions(
        self,
        image: torch.Tensor,
        regions: str,
        api_token: str,
        prompt_type: str = "text",
        text_prompt: str = "",
        return_caption: bool = True,
        return_roc: bool = False,
        return_ocr: bool = False,
    ) -> Tuple[str, str, str]:
        """
        Analyze image regions using DINO-X API

        Args:
            image: Input image tensor [B,H,W,C]
            regions: JSON array of region coordinates [[x1,y1,x2,y2],...]
            api_token: DINO-X API token
            prompt_type: Type of prompt ("text" or "universal")
            text_prompt: Text prompt for detection (required for "text" type)
            return_caption: Whether to return region captions
            return_roc: Whether to return ROC results
            return_ocr: Whether to return OCR results

        Returns:
            tuple: (captions, roc_results, ocr_results) as JSON strings
        """
        # Convert input tensor to OpenCV format
        image_bgr = self._prepare_image(image)

        # Save to temp file
        temp_path = self._save_temp_image(image_bgr)

        try:
            # Initialize API client
            config = Config(api_token)
            client = Client(config)

            # Parse regions
            import json

            regions_list = json.loads(regions)

            # Prepare targets
            targets = []
            if return_caption:
                targets.append("caption")
            if return_roc:
                targets.append("roc")
            if return_ocr:
                targets.append("ocr")

            # Prepare prompt
            if prompt_type == "text":
                prompt = {"type": "text", "text": text_prompt}
            else:
                prompt = {"type": "universal", "universal": 1}

            # Upload image and run analysis
            image_url = client.upload_file(temp_path)
            task = DinoxTask(
                model="DINO-X-1.0",  # Set model version
                image_url=image_url,
                prompt=prompt,
                regions=regions_list,
                targets=targets,
            )
            client.run_task(task)

            # Process results
            if not task.result or not hasattr(task.result, "objects"):
                return "{}", "{}", "{}"

            # Extract results by type
            captions = {}
            roc_results = {}
            ocr_results = {}

            for idx, obj in enumerate(task.result.objects):
                region_key = f"region_{idx}"
                region = obj.get("region", [])

                if return_caption:
                    captions[region_key] = {
                        "region": region,
                        "caption": obj.get("caption", ""),
                    }

                if return_roc:
                    roc_results[region_key] = {
                        "region": region,
                        "roc": obj.get("roc", ""),
                    }

                if return_ocr:
                    ocr_results[region_key] = {
                        "region": region,
                        "ocr": obj.get("ocr", ""),
                    }

            return (
                json.dumps(captions, indent=2),
                json.dumps(roc_results, indent=2),
                json.dumps(ocr_results, indent=2),
            )

        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Failed to clean up temp file: {str(e)}")


# Register nodes
NODE_CLASS_MAPPINGS = {
    "DinoxDetector": DinoxDetectorNode,
    "DinoxRegionVL": DinoxRegionVLNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DinoxDetector": "DINO-X Object Detector",
    "DinoxRegionVL": "DINO-X Region Analysis",
}
