# ComfyUI DINO-X Integration

A ComfyUI integration for DINO-X API, providing object detection, segmentation, pose estimation, and region-based analysis capabilities.

## Features

### DINO-X Object Detector Node

A node for object detection, segmentation, and keypoint estimation:

- Object detection with bounding boxes
- Instance segmentation masks
- Pose keypoint detection
- Hand keypoint detection
- Support for text and universal prompts
- Configurable confidence thresholds

#### Inputs
- `image`: Input image tensor [B,H,W,C]
- `text_prompt`: Text description of objects (e.g., "person . car . dog")
- `api_token`: DINO-X API token
- `bbox_threshold`: Detection confidence threshold (0.0-1.0)
- `iou_threshold`: IoU threshold for NMS (0.0-1.0)
- `prompt_type`: "text" or "universal"
- `return_pose`: Whether to return pose keypoints
- `return_hand`: Whether to return hand keypoints

#### Outputs
- `box_annotated`: Annotated image with boxes and labels [B,H,W,C]
- `binary_mask`: Binary segmentation mask [B,H,W]
- `pose_keypoints`: List of detected pose keypoints
- `hand_keypoints`: List of detected hand keypoints

### DINO-X Region Analysis Node

A node for analyzing specific regions in an image:

- Region-based captioning
- ROC (Region of Concern) analysis
- OCR text detection
- Support for up to 1000 regions
- Text and universal prompts

#### Inputs
- `image`: Input image tensor [B,H,W,C]
- `regions`: JSON array of region coordinates [[x1,y1,x2,y2],...]
- `api_token`: DINO-X API token
- `prompt_type`: "text" or "universal"
- `text_prompt`: Text prompt for detection
- `return_caption`: Whether to return captions
- `return_roc`: Whether to return ROC results
- `return_ocr`: Whether to return OCR results

#### Outputs
- `captions`: JSON string of region captions
- `roc_results`: JSON string of ROC analysis results
- `ocr_results`: JSON string of OCR results

## Installation

1. Get your DINO-X API token:
   - Visit [DeepDataSpace](https://cloud.deepdataspace.com/apply-token?from=github)
   - Register and request an API token
   - Save your token for use with the nodes

2. Install the nodes in your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-dinox-detector
cd comfyui-dinox-detector
pip install -r requirements.txt
```

## Usage Examples

### Object Detection

```python
# Load image → DINO-X Object Detector → Preview Image
{
    "3": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "example.jpg"
        }
    },
    "4": {
        "class_type": "DinoxDetector",
        "inputs": {
            "image": ["3", 0],
            "text_prompt": "person . car . dog",
            "api_token": "your-api-token-here",
            "bbox_threshold": 0.25,
            "iou_threshold": 0.8,
            "prompt_type": "text",
            "return_pose": true,
            "return_hand": false
        }
    },
    "5": {
        "class_type": "PreviewImage",
        "inputs": {
            "images": ["4", 0]  # Show box annotations
        }
    },
    "6": {
        "class_type": "PreviewImage",
        "inputs": {
            "images": ["4", 1]  # Show binary mask
        }
    }
}
```

### Region Analysis

```python
# Load image → DINO-X Region Analysis → Save Results
{
    "3": {
        "class_type": "LoadImage",
        "inputs": {
            "image": "example.jpg"
        }
    },
    "4": {
        "class_type": "DinoxRegionVL",
        "inputs": {
            "image": ["3", 0],
            "regions": "[[100,100,200,200], [300,300,400,400]]",
            "api_token": "your-api-token-here",
            "prompt_type": "text",
            "text_prompt": "person . car . dog",
            "return_caption": true,
            "return_roc": true,
            "return_ocr": true
        }
    }
}
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest dinox_detector/test_node.py -v
```

The tests cover:
- Input validation
- API interaction (mocked)
- Real-world image testing
- Tensor format handling
- Error handling

## License

This project is released under the Apache 2.0 license. See LICENSE file for details.
