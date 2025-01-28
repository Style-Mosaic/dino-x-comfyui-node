from .node import DinoxDetectorNode

NODE_CLASS_MAPPINGS = {"DinoxDetector": DinoxDetectorNode}

NODE_DISPLAY_NAME_MAPPINGS = {"DinoxDetector": "DINO-X Object Detector"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
