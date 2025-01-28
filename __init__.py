"""
DINO-X API integration for ComfyUI
"""

from .node import DinoxDetectorNode, DinoxRegionVLNode

# Register node classes
NODE_CLASS_MAPPINGS = {
    "DinoxDetector": DinoxDetectorNode,
    "DinoxRegionVL": DinoxRegionVLNode,
}

# Define display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "DinoxDetector": "DINO-X Object Detector",
    "DinoxRegionVL": "DINO-X Region Analysis",
}

# Required by ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("DINO-X nodes loaded:")
print(" - DinoxDetector: Object detection, segmentation, pose and hand keypoints")
print(" - DinoxRegionVL: Region-based analysis with captions, ROC, and OCR")
