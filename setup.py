from setuptools import find_packages, setup

setup(
    name="comfyui-dinox-detector",
    version="0.1.0",
    description="ComfyUI node for DINO-X API object detection and segmentation",
    author="Roo",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "supervision>=0.18.0",
        "dds-cloudapi-sdk>=0.3.3",
    ],
    python_requires=">=3.8",
)
