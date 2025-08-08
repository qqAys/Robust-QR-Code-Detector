#  Copyright (c) 2025 Jinx <jinx.timeless@gmail.com>.
#  Licensed under the MIT License.
#  See LICENSE file in the project root for full license information.

import cv2
import numpy as np
import os
import time
from math import sqrt
from PIL import Image, ImageDraw, ImageFont


class Config:
    """
    方便调试
    """

    # 图像预处理参数
    # 高斯模糊的卷积核大小，用于在轮廓查找前减少图像噪声。
    GAUSSIAN_BLUR_KERNEL = (3, 3)

    # 寻找定位点（Finder Pattern）的参数
    # 轮廓的最小面积，小于此值的轮廓将被忽略，以去除小的噪声点。
    CONTOUR_MIN_AREA = 20
    # `approxPolyDP`函数的近似多边形精度，用于将轮廓近似为多边形。
    # 轮廓周长的百分比，值越小，近似的多边形顶点越多。
    APPROX_POLY_EPSILON_FACTOR = 0.04
    # 宽高比容差，用于判断一个四边形轮廓是否接近正方形（QR码定位点的特征）。
    ASPECT_RATIO_TOLERANCE = 0.2

    # 寻找二维码区域的几何参数
    # 用于判断定位点之间距离是否相等（等腰直角三角形）的容差。
    DISTANCE_TOLERANCE = 0.2
    # 用于判断两个向量是否垂直（即直角）的点积容差。点积接近0则垂直。
    DOT_PRODUCT_TOLERANCE = 150

    # 透视变换和图像后处理参数
    # 透视变换后得到的二维码图像的边长（像素）。
    WARPED_QR_SIZE = 300
    # 锐化参数，`cv2.addWeighted`函数中的权重。
    SHARPEN_ALPHA = 0.8
    SHARPEN_BETA = -0.2
    # 亮度/对比度调整参数，`cv2.convertScaleAbs`函数中的参数。
    BRIGHTNESS_ALPHA = 0.2
    BRIGHTNESS_BETA = 8

    # 摄像头和保存设置
    # 摄像头的设备索引，通常0是默认摄像头。
    CAMERA_INDEX = 0
    # 摄像头捕获帧的宽度和高度。
    CAMERA_FRAME_WIDTH = 1280
    CAMERA_FRAME_HEIGHT = 720
    # 保存提取的二维码图像的文件夹名称。
    SAVE_FOLDER = "suspected_qr_codes"
    WINNAME = "Suspected QR Code"

    # 文本显示参数
    # 中文字体文件的路径。
    # 任意包含中文的字体均可，这里使用Noto Sans SC，可在此下载：https://fonts.google.com/noto/specimen/Noto+Sans+SC
    # 下载后解压，将.ttf文件放入脚本执行目录即可
    FONT_PATH = "NotoSansSC-VariableFont_wght.ttf"
    # 文本颜色，使用OpenCV的BGR格式。
    DISPLAY_TEXT_COLOR = (0, 0, 0)  # 黑色
    # 文本描边颜色。
    STROKE_COLOR = (255, 255, 255)  # 白色
    # 文本描边宽度。
    STROKE_WIDTH = 2
    # 显示文本的默认字号。
    DISPLAY_FONT_SIZE = 20


# 字体缓存。
_font_cache = {}


def get_font(font_size, font_path=Config.FONT_PATH):
    """
    获取指定大小的字体对象。
    - 如果字体已加载，则从`_font_cache`中获取。
    - 否则加载新字体并缓存。
    - 如果字体文件加载失败，会回退到默认字体并打印警告。
    """
    if font_size not in _font_cache:
        try:
            # 尝试加载字体。
            _font_cache[font_size] = ImageFont.truetype(font_path, font_size)
        except IOError:
            # 加载失败。
            print(
                f"警告: 无法加载字体文件 '{font_path}'，大小 {font_size}。中文文本可能无法正常显示。"
            )
            _font_cache[font_size] = ImageFont.load_default()
    return _font_cache[font_size]


def put_chinese_text(
    img,
    text,
    position,
    font_color,
    font_size=Config.DISPLAY_FONT_SIZE,
    stroke_color=Config.STROKE_COLOR,
    stroke_width=Config.STROKE_WIDTH,
):
    """
    使用PIL库在OpenCV图像上绘制中文文本及描边。
    - OpenCV不支持直接绘制中文字体，该函数通过PIL库进行桥接。
    """
    # 将OpenCV图像（BGR）转换为PIL图像（RGB）。
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    current_font = get_font(font_size)

    # 将OpenCV的BGR颜色元组转换为PIL的RGB颜色元组。
    text_fill = (font_color[2], font_color[1], font_color[0])
    stroke_fill = (stroke_color[2], stroke_color[1], stroke_color[0])

    # 绘制带描边的文本。
    draw.text(
        position,
        text,
        font=current_font,
        fill=text_fill,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )

    # 将绘制好的PIL图像（RGB）转换回OpenCV图像（BGR）。
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_overlay_text(
    img,
    messages,
    start_y,
    x_padding=15,
    line_spacing=10,
    color=Config.DISPLAY_TEXT_COLOR,
):
    """
    在图像上按行动态绘制多行文本。
    - 接受一个字符串列表`messages`，并从`start_y`位置开始，按行绘制。
    """
    y_offset = start_y
    for msg in messages:
        # 调用`put_chinese_text`绘制每一行文本。
        img = put_chinese_text(
            img, msg, (x_padding, y_offset), color, font_size=Config.DISPLAY_FONT_SIZE
        )
        # 更新Y坐标，为下一行文本留出空间。
        y_offset += Config.DISPLAY_FONT_SIZE + line_spacing
    return img


# --- 辅助函数：寻找定位点 ---
def find_qr_finder_patterns_optimized(frame):
    """
    寻找图像中潜在的二维码定位点（Finder Pattern）。
    - 这是一个优化后的版本，通过轮廓层级（hierarchy）来筛选，提高准确性。
    - QR码的定位点特征：一个由三个同心正方形组成的结构，在轮廓层级上表现为三层嵌套。
    - 返回每个定位点最外层轮廓的四个角点。
    """
    finder_patterns_points = []
    # 1. 图像预处理：
    # 将图像转换为灰度图。
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊，减少噪声。
    blurred = cv2.GaussianBlur(gray, Config.GAUSSIAN_BLUR_KERNEL, 0)
    # 2. 二值化：
    # 使用大津法（OTSU）进行二值化，自动确定最佳阈值。
    # `THRESH_BINARY`处理黑底白字的情况，`THRESH_BINARY_INV`处理白底黑字的情况。
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, thresh_inv = cv2.threshold(
        blurred, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # 遍历两种二值化结果，确保能检测到不同颜色模式的QR码。
    for thresh_img in [thresh, thresh_inv]:
        # 3. 查找轮廓：
        # `cv2.RETR_TREE`模式可以获取完整的轮廓层次结构（hierarchy）。
        contours, hierarchy = cv2.findContours(
            thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        # 4. 筛选轮廓：
        for i, contour in enumerate(contours):
            perimeter = cv2.arcLength(contour, True)
            # 使用`approxPolyDP`将轮廓近似为多边形，如果顶点数为4，则可能是正方形。
            approx = cv2.approxPolyDP(
                contour, Config.APPROX_POLY_EPSILON_FACTOR * perimeter, True
            )

            # 检查轮廓是否满足以下条件：
            if len(approx) == 4 and cv2.contourArea(contour) > Config.CONTOUR_MIN_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # 检查宽高比是否接近1（正方形）。
                if (
                    1 - Config.ASPECT_RATIO_TOLERANCE
                    <= aspect_ratio
                    <= 1 + Config.ASPECT_RATIO_TOLERANCE
                ):
                    # 检查轮廓的层级关系：
                    # 定位点是三层嵌套的正方形，因此最外层轮廓的父级是-1，且它有子轮廓。
                    parent = hierarchy[i][3]  # 获取当前轮廓的父级索引。
                    first_child = hierarchy[i][2]  # 获取当前轮廓的第一个子轮廓索引。

                    if parent == -1 and first_child != -1:
                        # 检查子轮廓是否有自己的子轮廓，构成三层嵌套。
                        inner_child = hierarchy[first_child][2]
                        if inner_child != -1:
                            # 满足所有条件，这是一个有效的定位点。
                            finder_patterns_points.append(approx.reshape(-1, 2))
    return finder_patterns_points


# --- 辅助函数：寻找二维码区域 ---
def find_qr_code_region(finder_patterns):
    """
    根据定位点（Finder Pattern）之间的几何关系，寻找二维码的四个角点。
    - 识别定位点是否构成一个等腰直角三角形，这是QR码的标志性特征。
    - 如果找到三个定位点，则通过几何计算确定第四个角点。
    - 如果只找到两个定位点，则假设它们是左上和右上，并推算出其他点。
    - 返回二维码的四个角点坐标，以及定位点的中心坐标。
    """
    if len(finder_patterns) < 2:
        return None, None

    centers = []
    # 1. 计算每个定位点的中心坐标（质心）。
    for pattern in finder_patterns:
        M = cv2.moments(pattern)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    if len(centers) < 2:
        return None, None

    qr_corners = None
    sorted_centers = None

    # 2. 如果找到三个或更多定位点（标准情况）：
    if len(centers) >= 3:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                for k in range(j + 1, len(centers)):
                    p1, p2, p3 = centers[i], centers[j], centers[k]

                    # 计算两点之间距离的平方。
                    d1_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                    d2_sq = (p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2
                    d3_sq = (p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2

                    distances_sq = sorted([d1_sq, d2_sq, d3_sq])

                    # 检查是否满足等腰直角三角形的条件：
                    # - 勾股定理：两短边平方和约等于长边平方。
                    # - 等腰：两短边长度约相等。
                    if (
                        abs(distances_sq[0] + distances_sq[1] - distances_sq[2])
                        / distances_sq[2]
                        < Config.DISTANCE_TOLERANCE
                        and abs(sqrt(distances_sq[0]) - sqrt(distances_sq[1]))
                        / sqrt(distances_sq[0])
                        < Config.DISTANCE_TOLERANCE
                    ):

                        # 检查哪个点是直角顶点，通过向量点积判断。点积接近0则垂直。
                        vector_p1_p2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                        vector_p1_p3 = np.array([p3[0] - p1[0], p3[1] - p1[1]])
                        vector_p2_p3 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

                        if (
                            abs(np.dot(vector_p1_p2, vector_p1_p3))
                            < Config.DOT_PRODUCT_TOLERANCE
                        ):
                            top_left_center, other_centers = p1, [p2, p3]
                        elif (
                            abs(np.dot(vector_p2_p3, -vector_p1_p2))
                            < Config.DOT_PRODUCT_TOLERANCE
                        ):
                            top_left_center, other_centers = p2, [p1, p3]
                        else:
                            top_left_center, other_centers = p3, [p1, p2]

                        # 确定另外两个点的方向（左上、右上、左下），通过向量叉积判断。
                        # 叉积符号决定了向量的相对方向。
                        vector1 = np.array(
                            [
                                other_centers[0][0] - top_left_center[0],
                                other_centers[0][1] - top_left_center[1],
                            ]
                        )
                        vector2 = np.array(
                            [
                                other_centers[1][0] - top_left_center[0],
                                other_centers[1][1] - top_left_center[1],
                            ]
                        )
                        cross_product = (
                            vector1[0] * vector2[1] - vector1[1] * vector2[0]
                        )

                        if cross_product > 0:
                            top_right_center, bottom_left_center = (
                                other_centers[0],
                                other_centers[1],
                            )
                        else:
                            top_right_center, bottom_left_center = (
                                other_centers[1],
                                other_centers[0],
                            )

                        # 根据中心点找到对应的定位点轮廓。
                        top_left_pattern = finder_patterns[
                            centers.index(top_left_center)
                        ]
                        top_right_pattern = finder_patterns[
                            centers.index(top_right_center)
                        ]
                        bottom_left_pattern = finder_patterns[
                            centers.index(bottom_left_center)
                        ]

                        # 找到每个定位点轮廓中最接近二维码实际角点的顶点。
                        # 左上角的角点是轮廓中x+y值最小的点。
                        top_left_qr_corner = sorted(
                            top_left_pattern, key=lambda p: p[0] + p[1]
                        )[0]
                        # 右上角的角点是轮廓中x-y值最大的点。
                        top_right_qr_corner = sorted(
                            top_right_pattern, key=lambda p: p[0] - p[1], reverse=True
                        )[0]
                        # 左下角的角点是轮廓中x-y值最小的点。
                        bottom_left_qr_corner = sorted(
                            bottom_left_pattern, key=lambda p: p[0] - p[1]
                        )[0]

                        # 通过向量加法推算出右下角点的位置。
                        # (右下) = (右上) + (左下) - (左上)
                        bottom_right_qr_corner_x = (
                            top_right_qr_corner[0]
                            + bottom_left_qr_corner[0]
                            - top_left_qr_corner[0]
                        )
                        bottom_right_qr_corner_y = (
                            top_right_qr_corner[1]
                            + bottom_left_qr_corner[1]
                            - top_left_qr_corner[1]
                        )
                        bottom_right_qr_corner = (
                            bottom_right_qr_corner_x,
                            bottom_right_qr_corner_y,
                        )

                        sorted_centers = [
                            top_left_center,
                            top_right_center,
                            bottom_left_center,
                        ]
                        return [
                            top_left_qr_corner,
                            top_right_qr_corner,
                            bottom_right_qr_corner,
                            bottom_left_qr_corner,
                        ], sorted_centers

    # 3. 如果只找到两个定位点（非标准情况）：
    elif len(centers) == 2:
        p1, p2 = centers[0], centers[1]
        pattern1, pattern2 = finder_patterns[0], finder_patterns[1]

        # 假设这两个点是左上和右上定位点，通过x+y值最小的作为左上角。
        if p1[0] + p1[1] < p2[0] + p2[1]:
            top_left_center = p1
            top_right_center = p2
            top_left_pattern = pattern1
            top_right_pattern = pattern2
        else:
            top_left_center = p2
            top_right_center = p1
            top_left_pattern = pattern2
            top_right_pattern = pattern1

        # 根据左上和右上两个中心点，推算出左下角的中心点。
        # 假设QR码是正方形，则左下角向量是右上角向量逆时针旋转90度。
        v_top_right = np.array(top_right_center) - np.array(top_left_center)
        v_bottom_left = np.array([-v_top_right[1], v_top_right[0]])
        bottom_left_center = tuple(np.array(top_left_center) + v_bottom_left)

        # 找到两个定位点轮廓的角点。
        top_left_qr_corner = sorted(top_left_pattern, key=lambda p: p[0] + p[1])[0]
        top_right_qr_corner = sorted(
            top_right_pattern, key=lambda p: p[0] - p[1], reverse=True
        )[0]

        # 根据角点推算出左下和右下角点。
        v_qr_top_right = np.array(top_right_qr_corner) - np.array(top_left_qr_corner)
        v_qr_bottom_left = np.array([-v_qr_top_right[1], v_qr_top_right[0]])
        bottom_left_qr_corner = tuple(np.array(top_left_qr_corner) + v_qr_bottom_left)
        bottom_right_qr_corner = tuple(np.array(top_right_qr_corner) + v_qr_bottom_left)

        sorted_centers = [top_left_center, top_right_center, bottom_left_center]
        return [
            top_left_qr_corner,
            top_right_qr_corner,
            bottom_right_qr_corner,
            bottom_left_qr_corner,
        ], sorted_centers

    return None, None


# --- 辅助函数：处理和保存二维码 ---
def process_qr_code(frame, qr_corners):
    """
    对找到的二维码区域进行透视变换、图像处理并保存。
    - 将倾斜或扭曲的QR码图像矫正为正方形。
    - 增强图像质量，使其更易于后续解码。
    """
    # 1. 透视变换：
    # `pts1`：原始图像中QR码四个角点的坐标。
    # `pts2`：目标图像中（正方形）的四个角点坐标。
    pts1 = np.float32(qr_corners)
    width, height = Config.WARPED_QR_SIZE, Config.WARPED_QR_SIZE
    pts2 = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )

    # 计算透视变换矩阵。
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 应用透视变换，将QR码区域矫正为正方形。
    warped_qr = cv2.warpPerspective(frame, matrix, (width, height))

    # 2. 图像后处理：
    # 转换为灰度图。
    gray_warped_qr = cv2.cvtColor(warped_qr, cv2.COLOR_BGR2GRAY)
    # 调整亮度和对比度。
    adjusted_qr = cv2.convertScaleAbs(
        gray_warped_qr, alpha=Config.BRIGHTNESS_ALPHA, beta=Config.BRIGHTNESS_BETA
    )
    # 对调整后的图像应用高斯模糊。
    blurred_qr = cv2.GaussianBlur(adjusted_qr, (0, 0), 3)
    # 锐化图像：通过将原图与模糊图加权叠加实现。
    sharpened_qr = cv2.addWeighted(
        adjusted_qr, Config.SHARPEN_ALPHA, blurred_qr, Config.SHARPEN_BETA, 0
    )
    # 再次使用大津法进行最终二值化，获得清晰的黑白图像。
    _, final_binary_qr = cv2.threshold(
        sharpened_qr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # 3. 确保保存文件夹存在。
    if not os.path.exists(Config.SAVE_FOLDER):
        os.makedirs(Config.SAVE_FOLDER)
    return final_binary_qr


def save_qr_code(final_binary_qr):
    """
    将处理后的二维码二值图像保存到文件。
    - 文件名包含当前时间戳，以确保唯一性。
    """
    # 构建文件名，包含文件夹、名称和时间戳。
    filename = f"{Config.SAVE_FOLDER}/qr_code_{int(time.time())}.png"
    # 将图像写入文件。
    cv2.imwrite(filename, final_binary_qr)
    print(f"二维码图片已保存至: {filename}")
    return filename


# --- 主循环 ---
def main():
    """
    主程序入口。
    - 初始化摄像头，进入主循环。
    - 实时捕捉帧，检测二维码区域。
    - 支持冻结、保存、匹配和退出等交互操作。
    """
    # 1. 初始化摄像头：
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    if not cap.isOpened():
        print("错误：无法打开摄像头。请检查摄像头连接或权限设置。")
        exit()

    # 设置摄像头捕获帧的尺寸。
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_FRAME_HEIGHT)

    print(" f : 冻结当前帧")
    print(" t : 旋转二维码方向 (仅在冻结状态下有效)")
    print(" c : 取消冻结，继续检测 (仅在冻结状态下有效)")
    print(" s : 保存疑似二维码 (仅在冻结状态下有效)")
    print(" q : 退出")

    is_frozen = False
    frozen_frame = None
    binary_qr_image = None
    last_saved_path = None
    detected_qr_corners = None
    detected_sorted_centers = None
    messages = []
    # 用于记录二维码角点的旋转状态
    qr_corner_rotation_index = 0

    # 2. 主循环：
    while True:
        # 如果未冻结，则实时捕获新帧。
        if not is_frozen:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            # 调用函数寻找定位点和二维码区域。
            finder_patterns_points = find_qr_finder_patterns_optimized(display_frame)
            qr_corners, sorted_centers = find_qr_code_region(finder_patterns_points)

            messages = []
            if qr_corners:
                # 找到二维码时，更新状态信息和绘制结果。
                messages.append(f"定位点 * {len(finder_patterns_points)}")

                # 添加定位点信息，在分辨率较高的时候会降低性能。
                # messages.append(f"{finder_patterns_points}")

                detected_qr_corners = qr_corners
                detected_sorted_centers = sorted_centers

                # 绘制二维码区域的绿色多边形边框。
                qr_corners_np = np.array(qr_corners, dtype=np.int32)
                cv2.polylines(
                    display_frame,
                    [qr_corners_np],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

                # 绘制定位点中心和连接线。
                if sorted_centers:
                    p1, p2, p3 = (
                        sorted_centers[0],
                        sorted_centers[1],
                        sorted_centers[2] if len(sorted_centers) > 2 else None,
                    )
                    cv2.line(display_frame, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)
                    if p3:
                        cv2.line(display_frame, p1, p3, (255, 0, 0), 1, cv2.LINE_AA)
                    for center in sorted_centers:
                        cv2.circle(
                            display_frame,
                            (int(center[0]), int(center[1])),
                            5,
                            (0, 0, 255),
                            -1,
                        )
            else:
                # 未找到二维码时，更新相应状态。
                messages.append("定位点 * 0")
                detected_qr_corners = None
                detected_sorted_centers = None
        else:
            # 如果已冻结，则显示冻结帧。
            display_frame = frozen_frame.copy()
            messages = ["已冻结"]

            # 在冻结帧上绘制上次检测到的二维码区域。
            if detected_qr_corners:
                # 根据当前的旋转索引，重新排列角点以进行绘制
                rotated_qr_corners_for_display = (
                    detected_qr_corners[qr_corner_rotation_index:]
                    + detected_qr_corners[:qr_corner_rotation_index]
                )
                qr_corners_np = np.array(rotated_qr_corners_for_display, dtype=np.int32)
                cv2.polylines(
                    display_frame,
                    [qr_corners_np],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
            if detected_sorted_centers:
                for center in detected_sorted_centers:
                    cv2.circle(
                        display_frame,
                        (int(center[0]), int(center[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )

        # 统一绘制底部和顶部的提示信息。
        if is_frozen:
            bottom_messages = ["操作: t: 旋转, c: 继续, s: 保存, q: 退出"]
        else:
            bottom_messages = ["操作: f: 冻结, q: 退出"]

        display_frame = draw_overlay_text(
            display_frame,
            bottom_messages,
            start_y=Config.CAMERA_FRAME_HEIGHT - Config.DISPLAY_FONT_SIZE - 20,
        )
        display_frame = draw_overlay_text(display_frame, messages, start_y=20)

        # 显示最终处理的图像。
        cv2.imshow("QR Code Region Detector", display_frame)

        # 3. 处理按键事件：
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            # 按'q'退出程序。
            break
        elif not is_frozen and key == ord("f") and detected_qr_corners:
            # 按'f'冻结帧，并处理检测到的二维码。
            frozen_frame = frame.copy()
            is_frozen = True
            qr_corner_rotation_index = 0
            # 对检测到的二维码区域进行透视变换和后处理。
            binary_qr_image = process_qr_code(frozen_frame, detected_qr_corners)
            # 显示处理后的二维码图像。
            cv2.destroyWindow(Config.WINNAME)
            cv2.imshow(Config.WINNAME, binary_qr_image)
        elif is_frozen and key == ord("c"):
            # 按'c'取消冻结，恢复实时检测。
            is_frozen = False
            frozen_frame = None
            detected_qr_corners = None
            detected_sorted_centers = None
            binary_qr_image = None
            qr_corner_rotation_index = 0
            cv2.destroyWindow(Config.WINNAME)
        elif is_frozen and key == ord("t") and detected_qr_corners:
            # 按't'旋转二维码方向
            qr_corner_rotation_index = (qr_corner_rotation_index + 1) % 4
            # 重新排列角点
            rotated_qr_corners = (
                detected_qr_corners[qr_corner_rotation_index:]
                + detected_qr_corners[:qr_corner_rotation_index]
            )
            # 使用新的角点重新处理图像
            binary_qr_image = process_qr_code(frozen_frame, rotated_qr_corners)
            # 更新显示
            cv2.imshow(Config.WINNAME, binary_qr_image)
        elif is_frozen and key == ord("s") and binary_qr_image is not None:
            # 按's'保存处理后的二维码。
            last_saved_path = save_qr_code(binary_qr_image)
            print(f"保存成功，文件路径: {last_saved_path}")

            # 这里原本是调用外部脚本进行二维码库匹配，因为要上传Github，所以移除了。
            # 你可以自行添加一些业务处理逻辑。

            # 保存成功后，自动取消冻结。
            is_frozen = False
            frozen_frame = None
            detected_qr_corners = None
            detected_sorted_centers = None
            binary_qr_image = None
            qr_corner_rotation_index = 0
            cv2.destroyWindow(Config.WINNAME)

    # 4. 退出前释放资源。
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
