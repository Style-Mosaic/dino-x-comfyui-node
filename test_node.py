import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from node import DinoxDetectorNode


@pytest.fixture
def mock_client():
    with patch("node.Client") as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def mock_config():
    with patch("node.Config") as mock:
        config_instance = Mock()
        mock.return_value = config_instance
        yield config_instance


@pytest.fixture
def synthetic_image():
    """Create a synthetic test image"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255  # White square
    return Image.fromarray(img)


@pytest.fixture
def real_image():
    """Load real test image"""
    return Image.open("test_leather_jacket.jpg")


@pytest.fixture
def comfyui_tensor_image():
    """Create a test image in ComfyUI tensor format (values in [0, 1])"""
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[30:70, 30:70] = 1.0  # White square
    return [img]  # ComfyUI sends as list of tensors


@pytest.fixture
def mock_predictions():
    return create_mock_predictions(100, 100)


@pytest.fixture
def mock_predictions_with_size(test_image):
    if isinstance(test_image, list):  # Handle ComfyUI tensor format
        h, w = test_image[0].shape[:2]
    else:
        w, h = test_image.size
    return create_mock_predictions(h, w)


def create_mock_predictions(height, width):
    prediction = Mock()
    prediction.bbox = [30, 30, 70, 70]
    prediction.mask = Mock()
    prediction.mask.counts = "test_rle_counts"
    prediction.mask.size = [height, width]
    prediction.score = 0.95
    prediction.category = "test_object"
    return [prediction]


@pytest.fixture(params=["synthetic", "real", "comfyui"])
def test_image(request, synthetic_image, real_image, comfyui_tensor_image):
    if request.param == "synthetic":
        return synthetic_image
    elif request.param == "real":
        return real_image
    else:
        return comfyui_tensor_image


def test_input_types():
    node = DinoxDetectorNode()
    input_types = node.INPUT_TYPES()

    assert "required" in input_types
    assert "image" in input_types["required"]
    assert "text_prompt" in input_types["required"]
    assert "api_token" in input_types["required"]
    assert "bbox_threshold" in input_types["required"]

    # Verify text_prompt configuration
    assert isinstance(input_types["required"]["text_prompt"], tuple)
    assert len(input_types["required"]["text_prompt"]) == 2
    assert input_types["required"]["text_prompt"][1]["multiline"] is True


def test_return_types():
    node = DinoxDetectorNode()

    assert node.RETURN_TYPES == ("IMAGE", "IMAGE")
    assert node.RETURN_NAMES == ("box_annotated", "binary_mask")
    assert node.CATEGORY == "detection"


def test_output_directory_creation():
    node = DinoxDetectorNode()
    assert Path("./outputs/dinox").exists()


@patch("node.DetectionTask")
def test_detect_and_annotate(
    mock_detection_task,
    mock_client,
    mock_config,
    test_image,
    mock_predictions_with_size,
):
    node = DinoxDetectorNode()

    # Mock API response
    mock_client.upload_file.return_value = "mock_url"
    task_instance = Mock()
    task_instance.result = Mock()
    task_instance.result.objects = mock_predictions_with_size
    mock_detection_task.return_value = task_instance

    # Mock RLE to mask conversion
    with patch("node.DetectionTask.rle2mask") as mock_rle2mask:
        with patch("node.DetectionTask.string2rle") as mock_string2rle:
            mock_rle2mask.return_value = np.ones(
                (
                    mock_predictions_with_size[0].mask.size[0],
                    mock_predictions_with_size[0].mask.size[1],
                ),
                dtype=bool,
            )
            mock_string2rle.return_value = "decoded_rle"

            # Run detection
            box_result, binary_mask = node.detect_and_annotate(
                test_image, "test_object", "test_token", 0.25
            )

    # Verify results
    assert isinstance(box_result, Image.Image)
    assert isinstance(binary_mask, Image.Image)
    assert binary_mask.mode == "L", "Binary mask should be in grayscale mode"

    if isinstance(test_image, list):  # ComfyUI tensor format
        h, w = test_image[0].shape[:2]
        assert box_result.size == (w, h)
        assert binary_mask.size == (w, h)
    else:
        assert box_result.size == test_image.size
        assert binary_mask.size == test_image.size


def test_leather_jacket_detection(real_image):
    node = DinoxDetectorNode()

    # Use real API with provided token
    try:
        box_result, binary_mask = node.detect_and_annotate(
            real_image,
            "leather jacket . person",
            "73b1fca55fffdf29a7f777d965bb64cf",
            0.25,
        )

        # Verify the results
        assert isinstance(box_result, Image.Image)
        assert isinstance(binary_mask, Image.Image)
        assert binary_mask.mode == "L", "Binary mask should be in grayscale mode"
        assert box_result.size == real_image.size
        assert binary_mask.size == real_image.size

        # Save test outputs for visual inspection
        Path("outputs").mkdir(parents=True, exist_ok=True)
        box_result.save("outputs/test_leather_jacket_box.jpg")
        binary_mask.save("outputs/test_leather_jacket_mask.png")

    except Exception as e:
        print(f"API test failed with error: {str(e)}")
        assert False, f"API test failed: {str(e)}"


def test_comfyui_tensor_input(comfyui_tensor_image):
    """Test that node can handle ComfyUI tensor input format"""
    node = DinoxDetectorNode()

    try:
        box_result, binary_mask = node.detect_and_annotate(
            comfyui_tensor_image,
            "test object",
            "73b1fca55fffdf29a7f777d965bb64cf",
            0.25,
        )

        assert isinstance(box_result, Image.Image)
        assert isinstance(binary_mask, Image.Image)
        assert binary_mask.mode == "L", "Binary mask should be in grayscale mode"
        h, w = comfyui_tensor_image[0].shape[:2]
        assert box_result.size == (w, h)
        assert binary_mask.size == (w, h)
    except Exception as e:
        assert False, f"Failed to process ComfyUI tensor input: {str(e)}"


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(threshold, test_image):
    node = DinoxDetectorNode()

    with pytest.raises(ValueError):
        node.detect_and_annotate(test_image, "test_object", "test_token", threshold)


def test_empty_api_token(test_image):
    node = DinoxDetectorNode()

    with pytest.raises(ValueError):
        node.detect_and_annotate(test_image, "test_object", "", 0.25)


def test_empty_text_prompt(test_image):
    node = DinoxDetectorNode()

    with pytest.raises(ValueError):
        node.detect_and_annotate(test_image, "", "test_token", 0.25)


def test_invalid_image_type():
    node = DinoxDetectorNode()
    invalid_image = "not_an_image"

    with pytest.raises(ValueError):
        node.detect_and_annotate(invalid_image, "test_object", "test_token", 0.25)


def test_cleanup_temp_files(test_image):
    node = DinoxDetectorNode()
    temp_path = Path("./outputs/dinox/temp_input.jpg")

    # Ensure clean state
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception as e:
            print(f"\nFailed to clean up existing temp file: {str(e)}")

    try:
        # Use real API with provided token
        box_result, binary_mask = node.detect_and_annotate(
            test_image, "test object", "73b1fca55fffdf29a7f777d965bb64cf", 0.25
        )

        # Verify mask is grayscale
        assert binary_mask.mode == "L", "Binary mask should be in grayscale mode"

        # Give the system a moment to complete file operations
        time.sleep(0.5)

        # Force Python to release file handles
        import gc

        gc.collect()

        # Check if file exists and get info
        if temp_path.exists():
            print(f"\nTemp file still exists at: {temp_path}")
            print(f"File size: {temp_path.stat().st_size} bytes")
            print(f"File permissions: {oct(temp_path.stat().st_mode)}")

        # Verify temp file is cleaned up
        assert not temp_path.exists(), "Temporary file was not cleaned up"

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise
    finally:
        # Clean up test outputs
        for f in Path("outputs").glob("test_*"):
            try:
                f.unlink(missing_ok=True)
            except Exception as e:
                print(f"\nFailed to clean up test output {f}: {str(e)}")
