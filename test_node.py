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
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255
    return Image.fromarray(img)


@pytest.fixture
def real_image():
    return Image.open("test_leather_jacket.jpg")


@pytest.fixture
def mock_predictions():
    return create_mock_predictions(100, 100)


@pytest.fixture
def mock_predictions_with_size(test_image):
    return create_mock_predictions(test_image.size[1], test_image.size[0])


def create_mock_predictions(height, width):
    prediction = Mock()
    prediction.bbox = [30, 30, 70, 70]
    prediction.mask = Mock()
    prediction.mask.counts = "test_rle_counts"
    prediction.mask.size = [height, width]
    prediction.score = 0.95
    prediction.category = "test_object"
    return [prediction]


@pytest.fixture(params=["synthetic", "real"])
def test_image(request, synthetic_image, real_image):
    if request.param == "synthetic":
        return synthetic_image
    else:
        return real_image


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
    assert node.RETURN_NAMES == ("box_annotated", "mask_annotated")
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
                (test_image.size[1], test_image.size[0]), dtype=bool
            )
            mock_string2rle.return_value = "decoded_rle"

            # Run detection
            box_result, mask_result = node.detect_and_annotate(
                test_image, "test_object", "test_token", 0.25
            )

    # Verify results
    assert isinstance(box_result, Image.Image)
    assert isinstance(mask_result, Image.Image)
    assert box_result.size == test_image.size
    assert mask_result.size == test_image.size


def test_leather_jacket_detection(real_image):
    node = DinoxDetectorNode()

    # Use real API with provided token
    try:
        box_result, mask_result = node.detect_and_annotate(
            real_image,
            "leather jacket . person",
            "73b1fca55fffdf29a7f777d965bb64cf",
            0.25,
        )

        # Verify the results
        assert isinstance(box_result, Image.Image)
        assert isinstance(mask_result, Image.Image)
        assert box_result.size == real_image.size
        assert mask_result.size == real_image.size

        # Save test outputs for visual inspection
        Path("outputs").mkdir(exist_ok=True)
        box_result.save("outputs/test_leather_jacket_box.jpg")
        mask_result.save("outputs/test_leather_jacket_mask.jpg")

    except Exception as e:
        print(f"API test failed with error: {str(e)}")
        assert False, f"API test failed: {str(e)}"


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

    with pytest.raises(TypeError):
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
        node.detect_and_annotate(
            test_image, "test object", "73b1fca55fffdf29a7f777d965bb64cf", 0.25
        )

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
