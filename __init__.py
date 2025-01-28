"""
DINO-X API integration for ComfyUI
"""

from .node import DinoxDetectorNode

# Register node class
NODE_CLASS_MAPPINGS = {"DinoxDetector": DinoxDetectorNode}

# Define display names
NODE_DISPLAY_NAME_MAPPINGS = {"DinoxDetector": "DINO-X Object Detector"}

# Required by ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
