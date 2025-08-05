# Contributing Guidelines

Thank you for considering contributing to the Vehicle Counter Flask App!


## How to Contribute

- Fork the repository and create your branch from `main`.
- If you've fixed a bug or added a feature, ensure relevant documentation and tests are updated.
- Open a pull request with a clear description of your changes.

## Model Weights (yolov8x.pt)

The `yolov8x.pt` model file is not included in this repository due to its size and licensing. If you do not have this file, you can download it from the official Ultralytics YOLOv8 release page:

- [YOLOv8 Releases on GitHub](https://github.com/ultralytics/ultralytics/releases)

Or, you can use the following Python code to download it automatically:

```python
from ultralytics import YOLO
YOLO('yolov8x.pt')  # This will download the model if not present
```

Place the downloaded `yolov8x.pt` file in the project root directory.

## Code Style

- Follow PEP8 for Python code style.
- Use clear variable and function names.
- Add docstrings to all public classes and functions.

## Issues

- If you find a bug, please open an issue with steps to reproduce.
- For feature requests, describe the motivation and possible alternatives.

## Community

- Be respectful and inclusive in all interactions.
- All contributions are welcome!
