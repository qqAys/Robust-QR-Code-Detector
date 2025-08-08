# Robust-QR-Code-Detector

[简体中文](./README.zh-CN.md)

A real-time QR code region detection tool based on `OpenCV`. It is designed to solve the problem of QR code recognition where finder patterns are missing due to physical damage or occlusion. Unlike traditional decoders, this project can deduce the complete QR code region even when only two finder patterns are detected, and it outputs a corrected, clear image, providing high-quality input for subsequent decoding.

![](./example.png)

## Core Features

- **Finder Pattern Detection**: Utilizes an optimized contour filtering algorithm to accurately identify QR code finder patterns within an image.

- **Geometric Deduction**: When the three finder patterns are incomplete, it can deduce the missing corner points using the existing ones and their geometric relationships.

- **Automatic Perspective Correction**: Corrects skewed and distorted QR code regions into a standard square image.

- **Image Enhancement**: Applies sharpening and adjusts brightness/contrast to the extracted QR code, improving readability.

- **Real-time Interactive Interface**: Captures footage from a camera in real-time and provides easy-to-use keyboard shortcuts for operation.

## Use Case

Recognizing QR codes with missing finder patterns due to physical damage or occlusion.

## Quick Start

### Dependencies

`OpenCV` (`cv2`)

`NumPy` (`numpy`)

`Pillow` (`PIL`)

### Installation

You can install all dependencies using `pip`：

```shell
pip install opencv-python numpy Pillow
```

### Running the Program

```shell
python main.py
```

### Instructions for Use

When the program starts, a camera window will open. The following keyboard shortcuts are available:

| Key | Function | Description |
| -------- | -------- | -------- |
|`f`|Freeze Frame|When a QR code is detected, press `f` to freeze the current frame.|
|`t`|Rotate Direction|While frozen, press `t` to rotate the detected QR code image.|
|`c`|Unfreeze|While frozen, press `c` to unfreeze and resume real-time detection.|
|`s`|Save Image|While frozen, press `s` to save the processed QR code image to your local drive.|
|`q`|Exit Program|Press `q` at any time to safely exit the program.|


## Brief Summary of Working Principle

1. **Contour Detection**: The input camera frame is binarized, and all contours are found using `cv2.findContours`.

2. **Finder Pattern Filtering**: Using the hierarchical relationships of contours obtained with `cv2.RETR_TREE mode`, concentric squares with a three-layer nested structure are filtered out. These are identified as potential QR code finder patterns.

3. **Geometric Analysis**:

    - **Three-Point Case**: If three finder patterns are found, the top-left, top-right, and bottom-left corner points are determined based on their relationship as an isosceles right triangle.

    - **Two-Point Case**: If only two finder patterns are found, they are assumed to be the top-left and top-right points, and the bottom-left and bottom-right corner points are deduced through vector rotation and addition.

4. **Perspective Transformation**: Using these four corner points as source coordinates, `cv2.getPerspectiveTransform` and `cv2.warpPerspective` are used to flatten and correct the QR code region.

5. **Image Optimization**: The corrected image is adjusted for brightness, contrast, and sharpened to obtain an optimal black-and-white binary image, which facilitates recognition by subsequent decoding libraries.

## Troubleshooting

### Displaying `????`
To correctly display Chinese text, you need to download a Chinese font file (e.g., [NotoSansSC-VariableFont_wght.ttf](https://fonts.google.com/noto/specimen/Noto+Sans+SC)) and place it in the project's root directory. If the font is not present, the program will automatically fall back to the default font, but Chinese characters may not display correctly.

### No Finder Patterns Detected

This can be influenced by various factors, such as the level of sharpening, motion blur, etc. You may need to manually adjust parameters or the algorithm to resolve this.

## License

This project is licensed under the MIT License.
