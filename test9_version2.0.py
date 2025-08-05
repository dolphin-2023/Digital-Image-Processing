import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve2d
import os
import glob
import struct
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import cv2  # OpenCV库用于HSV转换
import time # time库用来记录函数运行时间
#import cupy as cp # cupy库用于提高速度

# 配置 Matplotlib 字体，确保中文显示正常
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

#############################################
# 图像处理核心函数模块
#############################################
def read_raw_image(file_path):
    """
    读取自定义 RAW 格式图片。
    文件格式说明：
      - 前8个字节为文件头，分别存储宽度和高度（unsigned int，4字节×2）。
      - 剩余部分为图像像素数据，按uint8存储。
    """
    with open(file_path, 'rb') as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"文件头字节数错误：期望8字节，实际读取到{len(header)}字节")
        width, height = struct.unpack('II', header)
        pixels = f.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape((height, width))

def rgba_to_rgb(rgba_array, background=(1, 1, 1)):
    """ 将 RGBA 数组转换为 RGB（背景色归一化到 [0, 1]）"""
    alpha = rgba_array[..., 3:] / 255.0  # Alpha 通道 [0, 1]
    rgb = rgba_array[..., :3] / 255.0    # RGB 通道 [0, 1]
    bg = np.array(background)            # 背景色 (如 [1, 1, 1] 是白色)
    
    # Alpha 混合公式
    rgb_out = rgb * alpha + bg * (1 - alpha)
    return (rgb_out * 255).astype(np.uint8)  # 转回 [0, 255]

def create_pil_image(numpy_array):
    """
    将 NumPy 数组转换为 PIL Image 对象。
    根据数组维度判断是彩色图像还是灰度图像。
    """
    if len(numpy_array.shape) == 3:
        if numpy_array.shape[2] == 4:
            numpy_array = rgba_to_rgb(numpy_array)
        return Image.fromarray(numpy_array.astype(np.uint8), 'RGB')
    else:
        return Image.fromarray(numpy_array.astype(np.uint8), 'L')

# 图像处理相关函数
def invert_image(image):
    """
    反色处理（负片效果）：对图像每个像素进行 255 - 像素值的操作。
    为保证计算准确，先转换为浮点型再进行计算，最后还原为 uint8 类型。
    """
    img_float = image.astype(np.float32)
    inverted = 255 - img_float
    return np.clip(inverted, 0, 255).astype(np.uint8)

def log_transform_image(image, c=1.0):
    """
    对数变换增强图像。
    参数:
      image: 输入图像（uint8 类型）。
      c: 增强系数，建议范围1到5。
    过程:
      1. 将图像转换为浮点型。
      2. 对每个像素加1后取自然对数（使用 np.log1p 保证数值稳定性）。
      3. 根据最大对数值归一化并扩展到 [0,255] 范围，乘以增强系数。
      4. 结果限制在 0 到 255 之间，并转换为 uint8 类型。
    """
    img_float = image.astype(np.float32)
    log_image = np.log1p(img_float)  # 等价于 np.log(1 + image)
    max_val = np.log1p(255)  # 归一化最大值
    transformed = (log_image / max_val) * 255 * (c / 5)
    return np.clip(transformed, 0, 255).astype(np.uint8)

def _equalize_single_channel(channel):
    """
    对单通道图像进行直方图均衡化。
    过程：
      1. 使用 np.bincount 计算直方图。
      2. 计算累积分布函数（CDF）。
      3. 归一化 CDF 到 [0,255] 范围，并将原图像映射到新的灰度值。
    """
    hist = np.bincount(channel.ravel(), minlength=256) # .ravel是扁平化操作
    cdf = hist.cumsum() # 用于计算数组的 累积和（Cumulative Sum），即从左到右依次计算当前元素与之前所有元素的和（累积分布）
    cdf_nonzero = cdf[cdf > 0]
    if len(cdf_nonzero) == 0:
        return channel
    cdf_min = cdf_nonzero[0]
    cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255
    equalized = np.interp(channel.ravel(), np.arange(256), cdf_normalized).reshape(channel.shape)
    return equalized.astype(np.uint8)

def equalize_histogram(image):
    """
    对图像进行直方图均衡化以改善对比度。
    过程：
      - 灰度图像直接均衡化；
      - 彩色图像分别对 R、G、B 通道进行均衡化，再合并。
    """
    if len(image.shape) == 2:
        return _equalize_single_channel(image)
    elif len(image.shape) == 3:
        red_eq = _equalize_single_channel(image[:, :, 0])
        green_eq = _equalize_single_channel(image[:, :, 1])
        blue_eq = _equalize_single_channel(image[:, :, 2])
        return np.stack((red_eq, green_eq, blue_eq), axis=2)
    else:
        raise ValueError("不支持的图像格式")

def gamma_transform(image, gamma):
    """
    对图像进行伽马变换。
    过程：
      1. 将图像归一化到 [0,1]；
      2. 应用伽马公式：输出 = 输入^gamma；
      3. 将结果扩展回 [0,255] 并转换为 uint8 类型。
    """
    if image.ndim == 3:
        # 对每个通道分别处理
        channels = []
        for c in range(3):
            channel = image[:, :, c].astype(np.float32) / 255.0
            transformed = np.power(channel, gamma) * 255
            channels.append(transformed)
        return np.clip(np.stack(channels, axis=2), 0, 255).astype(np.uint8)
    else:
        # 灰度图像处理
        img_float = image.astype(np.float32) / 255.0
        transformed = np.power(img_float, gamma) * 255
        return np.clip(transformed, 0, 255).astype(np.uint8)

def calculate_gray_histogram(image_data):
    """
    计算灰度图像的归一化直方图。
    利用 np.bincount 统计每个灰度级出现次数，再除以总像素数获得概率分布。
    """
    counts = np.bincount(image_data.ravel(), minlength=256)
    return counts / image_data.size

def calculate_color_histogram(image_data):
    """
    计算彩色图像各通道的归一化直方图。
    分别计算 R、G、B 通道直方图并归一化。
    """
    red_channel = image_data[:, :, 0]
    green_channel = image_data[:, :, 1]
    blue_channel = image_data[:, :, 2]
    red_hist = np.bincount(red_channel.ravel(), minlength=256) / red_channel.size
    green_hist = np.bincount(green_channel.ravel(), minlength=256) / green_channel.size
    blue_hist = np.bincount(blue_channel.ravel(), minlength=256) / blue_channel.size
    return red_hist, green_hist, blue_hist

# 图像几何操作相关函数
def image_transpose(image):
    '''
    令图像围绕其中心逆时针旋转90度
    '''
    if image.ndim == 2:
        transposed = np.flipud(image.T)
    elif image.ndim == 3:
        transposed = np.flipud(np.swapaxes(image, 0, 1))
    else:
        raise ValueError("仅支持二维和三维图像")
    return transposed

def image_rotate(image, angle_degree=10):
    '''
    通用图像旋转函数
    默认一次逆时针转10°
    '''
    if image.ndim not in [2, 3]:
        raise ValueError("仅支持二维和三维图像")
    
    theta = np.deg2rad(-angle_degree) 
    # 之所以取负，是因为我底下函数用的逆变换。
    # 如果让函数"顺时针找源头"（参数用正角度），实际图像看起来会是逆时针转
    # 反之，如果让函数"逆时针找源头"（参数用负角度），图像看起来才是顺时针转

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 计算旋转后的新尺寸
    h, w = image.shape[:2]
    new_w = int(np.ceil(w*abs(cos_theta) + h*abs(sin_theta)))
    new_h = int(np.ceil(h*abs(cos_theta) + w*abs(sin_theta)))

    # 初始化输出图像
    if image.ndim == 3:
        rotated_img = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
    else:
        rotated_img = np.zeros((new_h, new_w), dtype=image.dtype)
    
    # 原图中心坐标
    cx = (w-1) / 2.0
    cy = (h-1) / 2.0
    # 新图中心坐标
    new_cx = (new_w - 1) / 2.0
    new_cy = (new_h - 1) / 2.0

    # 生成输出图像的坐标网络
    xs, ys = np.arange(new_w), np.arange(new_h)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    # 原点移至图像中心
    dx = xx - new_cx
    dy = yy - new_cy

    # 应用逆旋转矩阵
    x_rot = dx*cos_theta + dy*sin_theta
    y_rot = -dx*sin_theta + dy*cos_theta
    # 原点移至原图坐标系零点
    x_orig = x_rot + cx
    y_orig = y_rot + cy

    # 在原图范围内的坐标mask
    mask = (x_orig >= 0) & (x_orig <= w-1) & (y_orig >= 0) & (y_orig <= h-1)
    # 用np.clip限界
    x_orig = np.clip(x_orig, 0, w-1)
    y_orig = np.clip(y_orig, 0, h-1)

    # 双线性插值
    x0 = np.floor(x_orig).astype(int)
    y0 = np.floor(y_orig).astype(int)
    x1 = np.clip(x0+1, 0, w-1)
    y1 = np.clip(y0+1, 0, h-1)

    # 取小数部分
    dx = x_orig - x0
    dy = y_orig - y0

    # 处理彩色和灰度图像的不同情况
    if image.ndim == 3:
        # 彩色图像处理
        # 扩展dx和dy的维度以匹配彩色图像
        dx_3d = dx[..., np.newaxis]
        dy_3d = dy[..., np.newaxis]
        
        # 获取四个邻域像素值
        v00 = image[y0, x0, :]
        v10 = image[y0, x1, :]
        v01 = image[y1, x0, :]
        v11 = image[y1, x1, :]
        
        # 双线性插值计算
        interpolated = (
            (1 - dx_3d) * (1 - dy_3d) * v00 + 
            dx_3d * (1 - dy_3d) * v10 + 
            (1 - dx_3d) * dy_3d * v01 +
            dx_3d * dy_3d * v11
        )
    else:
        # 灰度图像处理
        v00 = image[y0, x0]
        v10 = image[y0, x1]
        v01 = image[y1, x0]
        v11 = image[y1, x1]
        
        interpolated = (
            (1 - dx) * (1 - dy) * v00 + 
            dx * (1 - dy) * v10 + 
            (1 - dx) * dy * v01 +
            dx * dy * v11
        )

    # 应用mask
    if image.ndim == 3:
        mask = mask[..., np.newaxis]
    rotated_img = (interpolated * mask).astype(image.dtype)
    
    return rotated_img

def image_translate(image, dx, dy):
    """
    图像平移函数
    参数：
        image: 输入图像矩阵（H,W）或（H,W,C）
        dx: 水平平移量（正数向右，负数向左）
        dy: 垂直平移量（正数向下，负数向上）
    返回：
        平移后的图像矩阵，超出部分填充0
    """
    output = np.zeros_like(image)
    h, w = image.shape[:2]
    src_x = slice(max(0, -dx), min(w, w - dx))
    src_y = slice(max(0, -dy), min(h, h - dy))
    dst_x = slice(max(0, dx), min(w, w + dx))
    dst_y = slice(max(0, dy), min(h, h + dy))
    output[dst_y, dst_x] = image[src_y, src_x]
    return output

def image_zoom_in(image, zoom_factor=2):
    """
    修复版图像放大函数（双线性插值实现）
    支持灰度图和彩色图，修复广播问题
    """
    if image.ndim not in [2, 3]:
        raise ValueError("仅支持二维和三维图像")

    h, w = image.shape[:2]
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)

    # 生成新坐标网格
    x = np.linspace(0, w-1, new_w) # 生成等间距的数值序列
    y = np.linspace(0, h-1, new_h)

    # 计算四个最近点的坐标
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, w-1)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, h-1)

    # 处理彩色图像
    if image.ndim == 3:
        # 使用高级索引获取像素块 (new_h, new_w, channels)
        v00 = image[y0[:, None], x0] # 广播，方便获取所有坐标值组合
        v10 = image[y0[:, None], x1]
        v01 = image[y1[:, None], x0]
        v11 = image[y1[:, None], x1]

        # 计算权重 (添加新轴以匹配彩色通道)
        # 为了确保权重能与之前获取的像素值正确广播相乘。
        dx = (x - x0).reshape(1, -1, 1)  # shape: (1, new_w, 1)
        dy = (y - y0).reshape(-1, 1, 1)  # shape: (new_h, 1, 1)

        # 双线性插值
        interpolated = ( (1-dy)*(1-dx)*v00 + 
                       (1-dy)*dx*v10 + 
                       dy*(1-dx)*v01 + 
                       dy*dx*v11 )
    
    # 处理灰度图像
    else:
        # 使用普通索引
        v00 = image[y0[:, None], x0]
        v10 = image[y0[:, None], x1]
        v01 = image[y1[:, None], x0]
        v11 = image[y1[:, None], x1]

        dx = (x - x0).reshape(1, -1)
        dy = (y - y0).reshape(-1, 1)

        interpolated = ( (1-dy)*(1-dx)*v00 + 
                       (1-dy)*dx*v10 + 
                       dy*(1-dx)*v01 + 
                       dy*dx*v11 )

    return np.clip(interpolated, 0, 255).astype(np.uint8)

def image_zoom_out(image, zoom_factor=0.5):
    """
    图像缩小函数（区域分块平均实现）- 格式固定为uint8
    """
    block_size = int(1 / zoom_factor) #每块的边长

    h, w = image.shape[:2]
    new_h = h // block_size
    new_w = w // block_size
    cropped = image[:new_h * block_size, :new_w * block_size] # 裁剪图像，确保尺寸能被 block_size 整除，我不想再插值了

    # reshape：例如，原图 (400, 600, 3)，block_size=2，则 reshape(200, 2, 300, 2, 3)，再对（2,2）块这个维度求均值
    if image.ndim == 3:
        zoomed = cropped.reshape(new_h, block_size, new_w, block_size, -1).mean(axis=(1, 3))
    else:
        zoomed = cropped.reshape(new_h, block_size, new_w, block_size).mean(axis=(1, 3))
    return np.clip(zoomed, 0, 255).astype(np.uint8)

# 基于阈值的图像分割相关函数
def otsu_threshold(image):
    """
    纯NumPy实现大津法阈值计算
    :param image: 输入灰度图像(0-255)
    :return: 最佳阈值
    """
    # 计算直方图
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # 归一化直方图(概率分布)
    hist_norm = hist / float(np.sum(hist))
    
    # 初始化最佳阈值和最大类间方差
    best_thresh = 0
    max_var = 0
    
    # 遍历所有可能的阈值(1-254，避免全部分到一类)
    for t in range(1, 255):
        # 将像素分为两类
        w0 = np.sum(hist_norm[:t])  # 背景权重
        w1 = np.sum(hist_norm[t:])  # 前景权重
        
        # 避免除零错误
        if w0 == 0 or w1 == 0:
            continue
        
        # 计算两类均值
        mu0 = np.sum(np.arange(t) * hist_norm[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist_norm[t:]) / w1
        
        # 计算类间方差
        var_between = w0 * w1 * (mu0 - mu1) ** 2
        
        # 更新最佳阈值
        if var_between > max_var:
            max_var = var_between
            best_thresh = t
    
    return best_thresh

def threshold_segmentation(image, threshold=None):
    """
    阈值分割 - 保留原始像素值，去掉小于阈值的像素
    参数:
        image: 输入图像(灰度或彩色)
        threshold: 可选阈值，None表示自动计算
    返回:
        分割后的图像(保持原图颜色/灰度)
    """
    if threshold is None:
        threshold = otsu_threshold(image)
    
    # 生成掩模（布尔矩阵）
    mask = image >= threshold
    
    # 对于彩色图像
    if len(image.shape) == 3:
        # 将掩模扩展到3个通道
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # 保留原图像素值
        result = np.where(mask, image, 0)
    else:
        # 灰度图像处理
        result = np.where(mask, image, 0)
    
    return result.astype(np.uint8)

def threshold_color_segmentation(image_rgb, target_color):
    """
    改进版彩色分割 - 优化色彩范围和后处理 - 用到了OpenCV
    参数:
        image_rgb: PIL.Image (RGB格式)
        target_color: 目标颜色 ('red', 'blue', 'yellow', 'green')
    返回:
        分割后的PIL.Image (RGB格式)
    """
    # 转换为OpenCV格式 (BGR)
    image = np.array(image_rgb)[:, :, ::-1]
    
    # 转换到HSV空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义各颜色的HSV范围
    color_ranges = {
        'red': [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),  # 红色范围1
            (np.array([160, 50, 50]), np.array([180, 255, 255])) # 红色范围2
        ],
        'blue': [
            (np.array([90, 50, 50]), np.array([130, 255, 255]))
        ],
        'yellow': [
            (np.array([20, 50, 50]), np.array([40, 255, 255]))
        ],
        'green': [
            (np.array([40, 50, 50]), np.array([80, 255, 255]))
        ]
    }
    
    if target_color.lower() not in color_ranges:
        raise ValueError(f"不支持的颜色 '{target_color}'，请选择: {list(color_ranges.keys())}")
    
    # 创建初始掩模
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # 应用所有范围
    for (lower, upper) in color_ranges[target_color.lower()]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    
    # 后处理优化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算填充小孔
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 开运算去除小噪点
    
    # 模糊边缘使过渡自然
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    
    # 应用掩模到原图
    mask = mask.astype(np.float32)/255.0
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    result = (image * mask).astype(np.uint8)
    
    # 转换回RGB格式
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

def hsv_value_threshold(image_rgb):
    """
    基于HSV Value通道的阈值分割（掩膜形式）
    参数:
        image_rgb: PIL.Image (RGB格式)
    返回:
        PIL.Image (RGB格式), 保留原图颜色的分割结果
    """
    # 将PIL图像转为NumPy数组(RGB)
    rgb_array = np.array(image_rgb)
    
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    
    # 提取Value通道
    v_channel = hsv[:, :, 2]
    
    # 使用大津法自动计算阈值
    threshold = otsu_threshold(v_channel)
    
    # 生成掩膜（布尔矩阵）
    mask = v_channel >= threshold
    
    # 将单通道掩膜扩展为3通道
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    
    # 应用掩膜保留原图颜色
    result = np.where(mask_3d, rgb_array, 0)
    
    return Image.fromarray(result.astype(np.uint8))

def canny_edge_segmentation(image, low_threshold=50, high_threshold=150):
    """
    Canny边缘分割方法
    参数:
        image: 输入图像(灰度或彩色)
        low_threshold: Canny低阈值
        high_threshold: Canny高阈值
    返回:
        边缘分割后的二值图像(边缘为白色255，背景为黑色0)
    """
    # 如果是彩色图像，先转换为灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges

# 基于边缘的图像分割相关函数
def Sobel_segmentation(image, auto_threshold=True, manual_threshold_ratio=0.2):
    """
    基于Sobel算子的边缘分割算法
    参数:
        image: 输入图像(灰度或彩色)
        threshold_ratio: 边缘强度阈值比例(0-1)
    返回:
        分割后的二值图像(边缘为白色255，背景为黑色0)
    """
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 定义Sobel核
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
    
    # 计算x和y方向的梯度
    grad_x = myconv2d(gray, sobel_x, padding=1)
    grad_y = myconv2d(gray, sobel_y, padding=1)
    
    # 计算梯度幅值
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化到0-255
    grad_magnitude = (grad_magnitude / np.max(grad_magnitude)) * 255
    
    # 自适应阈值处理（求大津阈值，方便做二值化）
    if auto_threshold:
        # 使用Otsu方法自动计算阈值
        threshold = otsu_threshold(grad_magnitude.astype(np.uint8))
    else:
        # 使用手动设置的阈值比例
        threshold = manual_threshold_ratio * np.max(grad_magnitude)
    
    # 二值化
    edges = np.where(grad_magnitude > threshold, 255, 0).astype(np.uint8)
    
    return edges

def Canny_segmentation(image, sigma=0.33, low_cut=24, high_cut=240, min_diff=30):
    """
    改进的自适应 Canny 边缘检测：忽略亮度极端值后再取 median
    参数:
        image: 输入图像 (灰度或彩色)
        sigma: 控制阈值范围的系数 (0.1~0.5)
        low_cut: 忽略的低亮度上限 (inclusive)
        high_cut: 忽略的高亮度下限 (inclusive)
        min_diff: 下阈值和上阈值的最小差值，防止断裂
    返回:
        边缘二值图 (白边黑底)
    """
    # 1. 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim==3 else image

    # 2. 先做轻微平滑，减少噪声对 std 的影响
    gray_blur = cv2.GaussianBlur(gray, (3,3), 0)
    
    # 3. 展平并剔除极端值
    '''
    如果图像背景大面积是纯黑（0）或纯白（255），
    直接对整个灰度图取 median，阈值就会被“拉”向极端，
    导致 Canny 检测效果变差
    '''
    flat = gray_blur.flatten()
    mask = (flat > low_cut) & (flat < high_cut)
    
    if mask.sum() > 0:
        median_val = np.median(flat[mask])
        std_val    = np.std(flat[mask])
    else:
        # 如果剔除后空了，就退回全量计算
        median_val = np.median(flat)
        std_val    = np.std(flat)

    # 4. 根据 median 和 std 计算动态阈值
    lower = int(max(0,   median_val * (1 - sigma) - 0.5 * std_val))
    upper = int(min(255, median_val * (1 + sigma) + 0.5 * std_val))
    # 保证差值足够
    if upper - lower < min_diff:
        upper = lower + min_diff
    
    # 5. 运行 Canny
    edges = cv2.Canny(gray_blur, lower, upper)
    return edges

# 图像滤波相关函数
def myconv2d(image, kernel, padding=0, stride=1):
    """
    优化后的二维卷积函数（使用向量化操作）
    参数:
        image: 输入图像(二维或三维数组)
        kernel: 卷积核(二维数组)
        padding: 零填充像素数
        stride: 卷积步长
    返回:
        卷积结果图像
    """
    # 记录开始时间
    start = time.perf_counter()

    # 检查输入有效性
    if len(image.shape) not in [2, 3]:
        raise ValueError("输入图像必须是二维或三维数组")
    if len(kernel.shape) != 2:
        raise ValueError("卷积核必须是二维数组")
    
    # 获取图像和卷积核尺寸
    kh, kw = kernel.shape
    if len(image.shape) == 2:
        h, w = image.shape
        channels = 1
        image = image[:, :, np.newaxis]  # 增加通道维度
    else:
        h, w, channels = image.shape
    
    # 计算输出尺寸
    out_h = (h - kh + 2 * padding) // stride + 1
    out_w = (w - kw + 2 * padding) // stride + 1
    
    # 添加零填充
    if padding > 0:
        padded = np.zeros((h + 2*padding, w + 2*padding, channels), dtype=image.dtype)
        padded[padding:-padding, padding:-padding, :] = image
    else:
        padded = image
    
    # 初始化输出
    output = np.zeros((out_h, out_w, channels), dtype=np.float32)
    
    # 预计算所有可能的窗口位置
    h_indices = np.arange(0, out_h) * stride
    w_indices = np.arange(0, out_w) * stride
    
    # 使用向量化操作计算卷积
    for c in range(channels):
        # 获取当前通道的图像数据
        channel_data = padded[:, :, c]
        
        # 创建所有可能的窗口视图
        windows = np.lib.stride_tricks.sliding_window_view(channel_data, (kh, kw))[::stride, ::stride]
        
        # 计算卷积结果
        output[:, :, c] = np.sum(windows * kernel, axis=(2, 3))
    
    # 移除单通道的额外维度
    if channels == 1:
        output = output.squeeze(axis=-1)
    
    # 记录结束时间
    end = time.perf_counter()
    print(f"myconv函数运行时间: {end - start:.6f} 秒")

    return output

def img_mean_filter(image, kernel_size=3):
    """
    均值滤波器
    参数:
        image: 输入图像
        kernel_size: 滤波器大小(奇数)
    返回:
        滤波后的图像
    """
    # 创建均值滤波器核
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # 应用卷积
    filtered = myconv2d(image, kernel, padding=kernel_size//2)
    
    # 确保输出数据类型与输入一致
    return np.clip(filtered, 0, 255).astype(image.dtype)

def img_Gaussian_filter(image, kernel_size=3, sigma=1.0):
    """
    高斯滤波器
    参数:
        image: 输入图像
        kernel_size: 滤波器大小(奇数)
        sigma: 高斯核标准差
    返回:
        滤波后的图像
    """
    # 创建高斯核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    # 计算高斯核值
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
    
    # 归一化核
    kernel /= np.sum(kernel)
    
    # 应用卷积
    filtered = myconv2d(image, kernel, padding=kernel_size//2)
    
    # 确保输出数据类型与输入一致
    return np.clip(filtered, 0, 255).astype(image.dtype)

def img_Sobel_filter(image, direction='both'):
    """
    Sobel边缘检测滤波器
    参数:
        image: 输入图像(灰度)
        direction: 检测方向('x', 'y'或'both')
    返回:
        边缘强度图像
    """
    # 检查输入是否为灰度图像
    if len(image.shape) != 2:
        raise ValueError("Sobel滤波器需要灰度图像输入")
    
    # 定义Sobel核
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # 计算x和y方向的梯度
    if direction in ['x', 'both']:
        grad_x = myconv2d(image, sobel_x, padding=1)
    else:
        grad_x = 0
    
    if direction in ['y', 'both']:
        grad_y = myconv2d(image, sobel_y, padding=1)
    else:
        grad_y = 0
    
    # 根据方向组合结果
    if direction == 'both':
        # 计算梯度幅值
        grad = np.sqrt(grad_x**2 + grad_y**2)
        # 归一化到0-255
        grad = (grad / np.max(grad)) * 255
    elif direction == 'x':
        grad = np.abs(grad_x)
        grad = (grad / np.max(grad)) * 255
    else:  # 'y'
        grad = np.abs(grad_y)
        grad = (grad / np.max(grad)) * 255
    
    return grad.astype(np.uint8)

def img_Laplace_filter(image):
    """
    Laplace边缘检测滤波器
    参数:
        image: 输入图像(灰度)
    返回:
        边缘强度图像
    """
    # 检查输入是否为灰度图像
    if len(image.shape) != 2:
        raise ValueError("Laplace滤波器需要灰度图像输入")
    
    # 定义Laplace核(4邻域)
    laplace_kernel = np.array([[0,  1, 0],
                              [1, -4, 1],
                              [0,  1, 0]])
    
    # 应用卷积
    edges = myconv2d(image, laplace_kernel, padding=1)
    
    # 取绝对值并归一化
    edges = np.abs(edges)
    edges = (edges / np.max(edges)) * 255
    
    return edges.astype(np.uint8)

def img_median_filter(image, kernel_size=3):
    """
    向量化实现的中值滤波器
    利用 numpy.lib.stride_tricks.sliding_window_view 消除所有显式循环
    参数:
        image: 输入图像(二维或三维数组)
        kernel_size: 滤波器大小(奇数)
    返回:
        滤波后的图像
    """
    # 记录开始时间
    start = time.perf_counter()

    # 检查输入有效性
    if len(image.shape) not in [2, 3]:
        raise ValueError("输入图像必须是二维或三维数组")
    
    # 处理单通道情况
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    h, w, channels = image.shape
    pad = kernel_size // 2
    
    # 零填充 (每个通道单独填充)
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    
    # 初始化输出
    output = np.zeros((h, w, channels), dtype=image.dtype)
    
    # 对每个通道进行向量化计算
    for c in range(channels):
        # 创建滑动窗口视图 (形状: [H, W, K, K])
        windows = sliding_window_view(padded[:, :, c], (kernel_size, kernel_size))
        
        # 计算每个窗口的中值 (axis=(2,3) 表示在最后两个维度上操作)
        output[:, :, c] = np.median(windows, axis=(2, 3))
    
    # 移除单通道的额外维度
    if squeeze_output:
        output = output.squeeze(axis=-1)
    
    # 记录结束时间
    end = time.perf_counter()
    print(f"向量化中值滤波函数运行时间: {end - start:.6f} 秒")

    return output

'''
def cupy_median_filter(image, kernel_size=3):
    """
    GPU加速的中值滤波器 (优化版)
    参数:
        image: 输入图像(二维或三维数组)
        kernel_size: 滤波器大小(奇数)
    返回:
        滤波后的图像 (numpy数组)
    """
    start = time.perf_counter()
    
    # 检查输入有效性
    if len(image.shape) not in [2, 3]:
        raise ValueError("输入图像必须是二维或三维数组")
    
    # 将数据从CPU迁移到GPU
    image_gpu = cp.asarray(image)
    
    # 处理单通道情况
    if len(image_gpu.shape) == 2:
        image_gpu = image_gpu[:, :, cp.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    h, w, channels = image_gpu.shape
    pad = kernel_size // 2
    
    # 零填充 (直接在GPU上操作)
    padded = cp.pad(image_gpu, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    
    # 初始化输出 (GPU内存)
    output_gpu = cp.zeros((h, w, channels), dtype=image_gpu.dtype)
    
    # 预计算偏移量
    offsets = cp.arange(-pad, pad + 1)
    xx, yy = cp.meshgrid(offsets, offsets)
    
    # 对每个像素和通道进行并行处理
    for c in range(channels):
        # 为当前通道创建所有可能的窗口
        windows = cp.zeros((h, w, kernel_size * kernel_size), dtype=image_gpu.dtype)
        
        # 收集所有邻居像素
        for i, (dx, dy) in enumerate(zip(xx.ravel(), yy.ravel())):
            windows[:, :, i] = padded[pad + dy : h + pad + dy, 
                                     pad + dx : w + pad + dx, c]
        
        # 计算中值
        output_gpu[:, :, c] = cp.median(windows, axis=2)
    
    # 移除单通道的额外维度
    if squeeze_output:
        output_gpu = output_gpu.squeeze(axis=-1)
    
    # 将结果移回CPU
    output = cp.asnumpy(output_gpu)

    end = time.perf_counter()
    print(f"优化后的CUDA中值滤波函数运行时间：{end - start:.6f} 秒")

    return output
'''

# 形态学处理相关函数
def binary_dilation(image, kernel_size=3, connectivity=8):
    """
    向量化实现的二值图像膨胀操作
    """
    if connectivity not in [4, 8]:
        raise ValueError("连通性必须是4或8")
    
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant')
    
    if connectivity == 8:
        # 使用滑动窗口视图
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        # 检查每个窗口中是否有255
        result = (np.max(windows, axis=(-2, -1)) > 0).astype(image.dtype) * 255
    else:
        # 4连通性可以分解为水平和垂直方向的分离操作
        kernel_h = np.zeros((1, kernel_size))
        kernel_h[0, :] = 1
        kernel_v = np.zeros((kernel_size, 1))
        kernel_v[:, 0] = 1
        
        dilated_h = (convolve2d(padded, kernel_h, mode='same') > 0).astype(image.dtype) * 255
        dilated_v = (convolve2d(padded, kernel_v, mode='same') > 0).astype(image.dtype) * 255
        result = np.maximum(dilated_h[pad:-pad, pad:-pad], dilated_v[pad:-pad, pad:-pad])
    
    return result

def binary_erosion(image, kernel_size=3, connectivity=8):
    """
    向量化实现的二值图像腐蚀操作
    """
    if connectivity not in [4, 8]:
        raise ValueError("连通性必须是4或8")
    
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant')
    
    if connectivity == 8:
        # 使用滑动窗口视图
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        # 检查每个窗口是否全部为255
        result = (np.min(windows, axis=(-2, -1)) > 0).astype(image.dtype) * 255
    else:
        # 4连通性可以分解为水平和垂直方向的分离操作
        kernel_h = np.zeros((1, kernel_size))
        kernel_h[0, :] = 1
        kernel_v = np.zeros((kernel_size, 1))
        kernel_v[:, 0] = 1
        
        eroded_h = (convolve2d(padded, kernel_h, mode='same') == kernel_size).astype(image.dtype) * 255
        eroded_v = (convolve2d(padded, kernel_v, mode='same') == kernel_size).astype(image.dtype) * 255
        result = np.minimum(eroded_h[pad:-pad, pad:-pad], eroded_v[pad:-pad, pad:-pad])
    
    return result

def binary_opening(image, kernel_size=3, connectivity=8):
    """开运算：先腐蚀后膨胀"""
    eroded = binary_erosion(image, kernel_size, connectivity)
    return binary_dilation(eroded, kernel_size, connectivity)

def binary_closing(image, kernel_size=3, connectivity=8):
    """闭运算：先膨胀后腐蚀"""
    dilated = binary_dilation(image, kernel_size, connectivity)
    return binary_erosion(dilated, kernel_size, connectivity)

def binary_gradient(image, kernel_size=3, connectivity=8):
    """形态学梯度：膨胀图减腐蚀图"""
    dilated = binary_dilation(image, kernel_size, connectivity)
    eroded = binary_erosion(image, kernel_size, connectivity)
    return dilated - eroded

def connected_component_labeling(image, connectivity=8):
    """
    连通域标记算法(使用scipy的现成实现)
    对于大型图像，两遍扫描法的纯Python实现仍然较慢，
    使用了scipy.ndimage.label或skimage.measure.label
    """
    from scipy.ndimage import label
    structure = np.ones((3,3)) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, _ = label(image//255, structure=structure)
    return labeled

'''
# 调用cuda前的预热
def warmup_gpu():
    """GPU预热函数"""
    warmup_data = cp.zeros((32, 32), dtype=cp.float32)
    _ = cp.median(warmup_data)  # 执行一个简单操作
    cp.cuda.Stream.null.synchronize()  # 等待操作完成
'''

class ImageViewerApp:
    def __init__(self, master):
        """初始化应用程序"""
        self.master = master
        # 初始化变量
        self._init_variables()
        # 设置UI
        self._setup_ui()
    
    def _init_variables(self):
        """初始化类变量"""
        self.image_paths = []
        self.current_index = -1
        self.original_image = None       # 原始图像数据
        self.base_processed_image = None # 基础处理结果
        self.processed_image = None      # 最终显示图像
        
        # 几何变换参数
        self.rotation_angle = 0         # 累计旋转角度
        self.scale_factor = 1.0         # 当前缩放比例
        self.translation = (0, 0)       # 平移量 (dx, dy)
        
        # 滤波器参数
        self.filter_kernel_size = 3  # 核大小默认值
        self.gaussian_sigma = 1.0    # 高斯滤波sigma默认值

        self.master.title("数字图像处理实验")
        self.master.geometry("1400x950")
        
        # 自定义样式
        self._configure_styles()

    def _configure_styles(self):
        """配置自定义样式"""
        style = ttk.Style()
        
        # 主框架样式
        style.configure('Main.TFrame', background='#f0f0f0')
        
        # 标签框架样式
        style.configure('Group.TLabelframe', 
                       padding=10, 
                       borderwidth=2, 
                       relief="groove",
                       background='#f0f0f0')
        style.configure('Group.TLabelframe.Label', 
                        font=('Arial', 10, 'bold'),
                        background='#f0f0f0')
        
        # 按钮样式
        style.configure('Tool.TButton', 
                       padding=5, 
                       font=('Arial', 9))
        
        # Notebook样式
        style.configure('TNotebook', padding=(5, 5, 5, 0))
        style.configure('TNotebook.Tab', 
                        padding=(10, 5), 
                        font=('Arial', 9, 'bold'))

    def _setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.master, style='Main.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部工具栏
        self._create_top_toolbar(main_frame)
        
        # 主内容区
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 图像显示区域
        self._create_image_display(content_frame)
        
        # 底部控制区
        self._create_bottom_controls(main_frame)
        
        # 信息显示区域
        self._create_info_display(main_frame)

    def _create_top_toolbar(self, parent):
        """创建顶部工具栏"""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 文件操作按钮
        file_buttons = [
            ("打开文件", self.open_file),
            ("打开目录", self.open_directory),
            ("保存原图", self.save_original_image),
            ("保存处理图", self.save_processed_image)
        ]
        
        for text, cmd in file_buttons:
            btn = ttk.Button(toolbar, text=text, command=cmd, style='Tool.TButton')
            btn.pack(side=tk.LEFT, padx=2)
        
        # 导航按钮
        nav_frame = ttk.Frame(toolbar)
        nav_frame.pack(side=tk.RIGHT, padx=10)
        
        ttk.Button(nav_frame, text="上一张", command=self.show_previous, style='Tool.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="下一张", command=self.show_next, style='Tool.TButton').pack(side=tk.LEFT, padx=2)

    def _create_image_display(self, parent):
        """创建图像显示区域"""
        img_frame = ttk.Frame(parent)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧原图面板
        left_frame = ttk.LabelFrame(img_frame, text="原图", style='Group.TLabelframe')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 原图操作按钮
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="原图逆时针旋转90°", command=self.transpose_original, style='Tool.TButton').pack(pady=2)
        
        # 原图画布
        self.image_canvas_left = tk.Canvas(left_frame, bg='#e0e0e0', highlightthickness=0)
        self.image_canvas_left.pack(fill=tk.BOTH, expand=True)
        # 当画布大小变化时，自动重绘原图
        self.image_canvas_left.bind(
            "<Configure>",
            lambda e: self.update_original_display()
        )

        # 右侧处理面板
        right_frame = ttk.LabelFrame(img_frame, text="处理后图像", style='Group.TLabelframe')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 处理后图像操作按钮
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        buttons = [
            ("逆时针旋转90°", self.transpose_processed),
            ("逆时针旋转10°", lambda: self.rotate_processed(10)),
            ("平移", self.translate_processed),
            ("放大两倍", lambda: self.zoom_processed('in')),
            ("缩小1/2", lambda: self.zoom_processed('out')),
            ("恢复", self.reset_processed_image)
        ]
        
        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd, style='Tool.TButton').pack(side=tk.LEFT, padx=2)
        
        # 处理后图像画布（带滚动条）
        container = ttk.Frame(right_frame)
        container.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas_right = tk.Canvas(container, bg='#e0e0e0', highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.image_canvas_right.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=self.image_canvas_right.xview)
        
        self.image_canvas_right.configure(
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set
        )
        
        # 布局
        self.image_canvas_right.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

    def _create_bottom_controls(self, parent):
        """创建底部控制区"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 使用Notebook实现选项卡式界面
        control_notebook = ttk.Notebook(control_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 选项卡1：基础处理
        basic_tab = ttk.Frame(control_notebook)
        self._create_basic_controls(basic_tab)
        control_notebook.add(basic_tab, text="基础处理")
        
        # 选项卡2：几何变换
        geo_tab = ttk.Frame(control_notebook)
        self._create_geometric_controls(geo_tab)
        control_notebook.add(geo_tab, text="几何变换")
        
        # 选项卡3：图像滤波
        filter_tab = ttk.Frame(control_notebook)
        self._create_filter_controls(filter_tab)
        control_notebook.add(filter_tab, text="图像滤波")
        
        # 选项卡4：图像分割
        segment_tab = ttk.Frame(control_notebook)
        self._create_segmentation_controls(segment_tab)
        control_notebook.add(segment_tab, text="图像分割")

        # 选项卡5：形态学处理
        morph_tab = ttk.Frame(control_notebook)
        self._create_morphology_controls(morph_tab)
        control_notebook.add(morph_tab, text="形态学处理")

    def _create_basic_controls(self, parent):
        """创建基础处理控制"""
        frame = ttk.LabelFrame(parent, text="基础图像处理", style='Group.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 基本处理按钮
        buttons = [
            ("反色", self.apply_invert),
            ("对数变换", self.apply_log_transform),
            ("直方图均衡", self.apply_histogram_equalization)
        ]
        
        for text, cmd in buttons:
            ttk.Button(frame, text=text, command=cmd, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        
        # 伽马变换控制
        gamma_frame = ttk.Frame(frame)
        gamma_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.gamma_label = ttk.Label(gamma_frame, text="幂次变换 (1.00)")
        self.gamma_label.pack(anchor='w')
        
        self.gamma_slider = ttk.Scale(
            gamma_frame, from_=0.1, to=5.0, 
            orient=tk.HORIZONTAL, length=200,
            command=self.update_gamma_preview
        )
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(fill=tk.X)

    def _create_geometric_controls(self, parent):
        """创建几何变换控制"""
        frame = ttk.LabelFrame(parent, text="几何变换参数", style='Group.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 这里可以添加几何变换的特定参数控制
        ttk.Label(frame, text="几何变换已整合到图像显示区域的按钮").pack(pady=20)

    def _create_filter_controls(self, parent):
        """创建滤波器控制"""
        frame = ttk.LabelFrame(parent, text="滤波器参数", style='Group.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 滤波器按钮
        filter_buttons = [
            ("均值滤波", self.apply_mean_filter),
            ("中值滤波", self.apply_median_filter),
            ("高斯滤波", self.apply_gaussian_filter),
            ("Sobel滤波", self.apply_sobel_filter),
            ("Laplace滤波", self.apply_laplace_filter)
        ]
        
        for text, cmd in filter_buttons:
            ttk.Button(frame, text=text, command=cmd, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        
        # 滤波器参数控制
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        # 核大小滑动条
        ttk.Label(param_frame, text="核大小:").pack(side=tk.LEFT)
        self.kernel_size_slider = ttk.Scale(
            param_frame, from_=3, to=15, 
            orient=tk.HORIZONTAL, length=100
        )
        self.kernel_size_slider.set(3)
        self.kernel_size_slider.pack(side=tk.LEFT, padx=5)
        self.kernel_size_label = ttk.Label(param_frame, text="3")
        self.kernel_size_label.pack(side=tk.LEFT)
        self.kernel_size_slider.configure(command=self.update_kernel_size)
        
        # 高斯滤波sigma值滑动条
        ttk.Label(param_frame, text="Sigma:").pack(side=tk.LEFT, padx=(10,0))
        self.sigma_slider = ttk.Scale(
            param_frame, from_=0.1, to=5.0, 
            orient=tk.HORIZONTAL, length=100
        )
        self.sigma_slider.set(1.0)
        self.sigma_slider.pack(side=tk.LEFT, padx=5)
        self.sigma_label = ttk.Label(param_frame, text="1.0")
        self.sigma_label.pack(side=tk.LEFT)
        self.sigma_slider.configure(command=self.update_sigma_value)

    def _create_segmentation_controls(self, parent):
        """创建图像分割控制"""
        frame = ttk.LabelFrame(parent, text="图像分割", style='Group.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 分割方法按钮
        ttk.Button(frame, text="Otsu阈值法分割", command=self.apply_threshold_segmentation, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frame, text="Sobel边缘检测", command=self.apply_sobel_segmentation, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frame, text="基于阈值的Canny", command=self.apply_canny_segmentation, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frame, text="Canny边缘检测", command=self.apply_auto_canny_segmentation, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)

        # 色彩分割控制
        color_frame = ttk.Frame(frame)
        color_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.color_var = tk.StringVar(value="red")
        color_menu = ttk.OptionMenu(
            color_frame, self.color_var, "red", 
            "red", "blue", "yellow", "green"
        )
        color_menu.pack(side=tk.LEFT, padx=2)
        ttk.Button(color_frame, text="色彩分割", command=self.apply_color_segmentation, style='Tool.TButton').pack(side=tk.LEFT, padx=2)

    def _create_morphology_controls(self, parent):
        """创建形态学处理控制界面"""
        frame = ttk.LabelFrame(parent, text="形态学操作", style='Group.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 操作按钮
        morph_buttons = [
            ("膨胀", self.apply_dilation),
            ("腐蚀", self.apply_erosion),
            ("开运算", self.apply_opening),
            ("闭运算", self.apply_closing),
            ("形态学梯度", self.apply_gradient),
            ("连通域标记", self.apply_labeling)
        ]
        
        for text, cmd in morph_buttons:
            ttk.Button(frame, text=text, command=cmd, style='Tool.TButton').pack(side=tk.LEFT, padx=5, pady=5)
        
        # 参数控制
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        # 核大小滑动条
        ttk.Label(param_frame, text="核大小:").pack(side=tk.LEFT)
        self.morph_kernel_size = ttk.Scale(
            param_frame, from_=3, to=15, 
            orient=tk.HORIZONTAL, length=100
        )
        self.morph_kernel_size.set(3)
        self.morph_kernel_size.pack(side=tk.LEFT, padx=5)
        self.morph_kernel_label = ttk.Label(param_frame, text="3")
        self.morph_kernel_label.pack(side=tk.LEFT)
        self.morph_kernel_size.configure(command=self.update_morph_kernel_size)
        
        # 连通性选择
        ttk.Label(param_frame, text="连通性:").pack(side=tk.LEFT, padx=(10,0))
        self.connectivity_var = tk.StringVar(value="8")
        ttk.Radiobutton(param_frame, text="4-邻域", variable=self.connectivity_var, value="4").pack(side=tk.LEFT)
        ttk.Radiobutton(param_frame, text="8-邻域", variable=self.connectivity_var, value="8").pack(side=tk.LEFT)

    def _create_info_display(self, parent):
        """创建信息显示区域"""
        # 图片信息标签
        self.info_label = ttk.Label(parent, text="图片信息：未加载")
        self.info_label.pack(pady=5)
        
        # 直方图显示
        hist_frame = ttk.LabelFrame(parent, text="直方图", style='Group.TLabelframe')
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.figure = Figure(figsize=(10, 2)) # 10英寸宽，2英寸高
        self.hist_canvas = FigureCanvasTkAgg(self.figure, hist_frame) # 将Matplotlib图形嵌入Tkinter
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ============== 图像显示更新方法 ==============
    def update_original_display(self):
        """更新原图显示"""
        if self.original_image is None:
            self.image_canvas_left.delete("all") # 清空画布
            self.image_canvas_left.image = None # 清除引用
            return
        try: # 添加 try-except 块捕获可能的错误        
            pil_image = create_pil_image(self.original_image)
            canvas = self.image_canvas_left

            # 为避免自适应问题，强制完成一次布局，再读真实尺寸
            canvas.update_idletasks()
            canvas_width  = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            # 防止除零或尺寸无效
            if canvas_width <= 0 or canvas_height <= 0:
                 canvas_width = 200 # 设定一个默认最小尺寸
                 canvas_height = 200
            if pil_image.width <= 0 or pil_image.height <= 0:
                 print("Error: Original PIL image has zero dimension.")
                 return # 避免后续错误

            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            new_size = (int(img_width * scale), int(img_height * scale))

            # --- 尝试修改这里 ---
            # 尝试不同的缩放算法，比如 NEAREST，看是否能正确显示
            # new_size = (int(img_width * scale), int(img_height * scale))
            resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            # resized_image = pil_image.resize(new_size, Image.NEAREST)
            # resized_image = pil_image.resize(new_size, Image.BILINEAR)
            # resized_image = pil_image.resize(new_size, Image.BICUBIC)
            # resized_image = pil_image.resize(new_size, Image.BOX)
            # resized_image = pil_image.resize(new_size, Image.Resampling.HAMMING)
            # resized_image = pil_image   # 我错了，我再也不玩重采样/缩放了
            # new_size = (img_width, img_height) # new_size 只用于定位

            tk_image = ImageTk.PhotoImage(resized_image)
            
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=tk_image,
                anchor=tk.CENTER
            )
            canvas.image = tk_image  # 保持引用

        except Exception as e:
            print(f"Error during update_original_display: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误信息
            # 出错时尝试清空画布
            self.image_canvas_left.delete("all")
            self.image_canvas_left.image = None
    
    def update_processed_display(self):
        """更新处理后图像显示"""
        if self.processed_image is None:
            return
            
        pil_image = create_pil_image(self.processed_image)
        img_width, img_height = pil_image.size
        
        # 获取原图尺寸作为参考
        orig_width, orig_height = (img_width, img_height)
        if self.original_image is not None:
            orig_pil = create_pil_image(self.original_image)
            orig_width, orig_height = orig_pil.size
        
        # 创建足够大的画布区域
        canvas_width = max(img_width, orig_width)
        canvas_height = max(img_height, orig_height)
        
        # 创建背景并居中显示图像
        background = Image.new("RGB", (canvas_width, canvas_height), "black")
        paste_x = (canvas_width - img_width) // 2
        paste_y = (canvas_height - img_height) // 2
        background.paste(pil_image, (paste_x, paste_y))
        
        # 更新画布
        tk_image = ImageTk.PhotoImage(background)
        canvas = self.image_canvas_right
        
        canvas.delete("all")
        canvas.config(
            scrollregion=(0, 0, canvas_width, canvas_height),
            width=min(canvas_width, 800),
            height=min(canvas_height, 600)
        )
        
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=tk_image)
        canvas.image = tk_image  # 保持引用
        
        # 自动滚动到中心
        canvas.xview_moveto(0.5 - (400/canvas_width))
        canvas.yview_moveto(0.5 - (300/canvas_height))
    
    def update_info_label(self):
        """更新图片信息标签"""
        if self.original_image is None:
            return
            
        if len(self.original_image.shape) == 2:
            height, width = self.original_image.shape
            channels = 1
        else:
            height, width, channels = self.original_image.shape
            
        info_text = f"尺寸：{width}×{height} 像素 | 通道：{channels} | 当前：{self.current_index + 1}/{len(self.image_paths)}"
        self.info_label.config(text=info_text)
    
    def update_histogram(self):
        """更新直方图显示"""
        self.figure.clear()
        
        # 原图直方图
        ax1 = self.figure.add_subplot(121) # 一行二列第一个子图
        if self.original_image is not None:
            if len(self.original_image.shape) == 2:
                hist = calculate_gray_histogram(self.original_image)
                ax1.bar(range(256), hist, width=1.0, color='gray')
            else:
                r, g, b = calculate_color_histogram(self.original_image)
                ax1.plot(range(256), r, color='red')
                ax1.plot(range(256), g, color='green')
                ax1.plot(range(256), b, color='blue')
            ax1.set_title("原图直方图")
        
        # 处理后图像直方图
        ax2 = self.figure.add_subplot(122)
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:
                hist = calculate_gray_histogram(self.processed_image)
                ax2.bar(range(256), hist, width=1.0, color='gray')
            else:
                r, g, b = calculate_color_histogram(self.processed_image)
                ax2.plot(range(256), r, color='red')
                ax2.plot(range(256), g, color='green')
                ax2.plot(range(256), b, color='blue')
            ax2.set_title("处理后直方图")
        
        self.figure.tight_layout() # 自动调整子图间距
        self.hist_canvas.draw() # 更新画布显示

    # ============== 图像处理方法 ==============
    def apply_invert(self):
        """应用反色处理"""
        if self.original_image is not None:
            self.base_processed_image = invert_image(self.original_image)
            self._apply_geometric_transforms()
    
    def apply_log_transform(self):
        """应用对数变换"""
        if self.original_image is None:
            return
            
        c = simpledialog.askfloat(
            "对数变换", "请输入增强系数(范围1-5)：", 
            parent=self.master, minvalue=1.0, maxvalue=5.0
        )
        
        if c is not None:
            self.base_processed_image = log_transform_image(self.original_image, c)
            self._apply_geometric_transforms()
    
    def apply_histogram_equalization(self):
        """应用直方图均衡化"""
        if self.original_image is not None:
            self.base_processed_image = equalize_histogram(self.original_image)
            self._apply_geometric_transforms()
    
    def update_gamma_preview(self, value):
        """更新伽马变换预览"""
        if self.original_image is not None:
            gamma = float(value)
            self.gamma_label.config(text=f"幂次变换 ({gamma:.2f})")
            self.base_processed_image = gamma_transform(self.original_image, gamma)
            self._apply_geometric_transforms()

    def _update_processed_image(self, image):
        """更新处理后图像(内部方法)"""
        self.base_processed_image = image
        self.processed_image = image.copy()
        self.reset_geometric_params()
        self.update_processed_display()
        self.update_histogram()

    # ============== 图像分割方法 ==============    
    def apply_threshold_segmentation(self):
        """应用阈值分割"""
        if self.original_image is None:
            return
            
        pil_image = create_pil_image(self.original_image)
        
        if len(self.original_image.shape) == 2:
            segmented = threshold_segmentation(self.original_image)
        else:
            segmented = hsv_value_threshold(pil_image)
        
        self._update_processed_image(np.array(segmented))
    
    def apply_canny_segmentation(self):
        """应用Canny边缘分割"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        
        # 创建阈值设置对话框
        class CannyThresholdDialog(simpledialog.Dialog):
            def __init__(self, parent, title=None):
                self.low_threshold = 50
                self.high_threshold = 150
                super().__init__(parent, title)
            
            def body(self, master):
                ttk.Label(master, text="低阈值(0-255):").grid(row=0, sticky=tk.W)
                ttk.Label(master, text="高阈值(0-255):").grid(row=1, sticky=tk.W)
                
                self.low_entry = ttk.Entry(master)
                self.high_entry = ttk.Entry(master)
                
                self.low_entry.grid(row=0, column=1)
                self.high_entry.grid(row=1, column=1)
                
                self.low_entry.insert(0, str(self.low_threshold))
                self.high_entry.insert(0, str(self.high_threshold))
                
                return self.low_entry
            
            def validate(self):
                try:
                    self.low_threshold = int(self.low_entry.get())
                    self.high_threshold = int(self.high_entry.get())
                    
                    if not (0 <= self.low_threshold <= 255):
                        raise ValueError("低阈值必须在0-255之间")
                    if not (0 <= self.high_threshold <= 255):
                        raise ValueError("高阈值必须在0-255之间")
                    if self.low_threshold >= self.high_threshold:
                        raise ValueError("低阈值必须小于高阈值")
                    
                    return True
                except ValueError as e:
                    messagebox.showerror("输入错误", str(e))
                    return False
        
        # 显示阈值设置对话框
        dialog = CannyThresholdDialog(self.master, "设置Canny阈值")
        
        # 如果用户确认了阈值设置
        if dialog.low_threshold is not None and dialog.high_threshold is not None:
            # 应用Canny边缘分割
            edges = canny_edge_segmentation(
                self.original_image, 
                dialog.low_threshold, 
                dialog.high_threshold
            )
            
            # 将二值图像转换为RGB格式以便显示（边缘为白色，背景为黑色）
            if len(self.original_image.shape) == 3:
                # 如果是彩色图像，创建红色边缘效果
                edge_rgb = np.zeros_like(self.original_image)
                edge_rgb[edges > 0] = [255, 0, 0]  # 边缘设为红色
            else:
                # 如果是灰度图像，创建白色边缘效果
                edge_rgb = np.zeros((*edges.shape, 3), dtype=np.uint8)
                edge_rgb[edges > 0] = [255, 255, 255]  # 边缘设为白色
            
            self._update_processed_image(edge_rgb)

    def apply_color_segmentation(self):
        """应用色彩分割"""
        if self.original_image is None or len(self.original_image.shape) != 3:
            messagebox.showwarning("提示", "请先打开彩色图像")
            return
            
        target_color = self.color_var.get()
        pil_image = create_pil_image(self.original_image)
        
        try:
            segmented = threshold_color_segmentation(pil_image, target_color)
            self._update_processed_image(np.array(segmented))
        except ValueError as e:
            messagebox.showerror("错误", str(e))
    
    def apply_sobel_segmentation(self):
        """应用Sobel边缘分割"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        
        # 更新处理后图像
        edges = Sobel_segmentation(self.original_image)
        self._update_processed_image(edges)

    def apply_auto_canny_segmentation(self):
        """应用自动Canny边缘分割"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        
        # 更新处理后图像
        edges = Canny_segmentation(self.original_image)
        self._update_processed_image(edges)

    # ============== 几何变换方法 ==============
    def reset_geometric_params(self):
        """重置几何变换参数"""
        self.rotation_angle = 0
        self.scale_factor = 1.0
        self.translation = (0, 0)
    
    def _apply_geometric_transforms(self):
        """应用所有几何变换"""
        if self.base_processed_image is None:
            return
            
        img = self.base_processed_image.copy()
        
        # 1. 缩放
        if self.scale_factor != 1.0:
            try:
                if self.scale_factor > 1:
                    img = image_zoom_in(img, self.scale_factor)
                else:
                    img = image_zoom_out(img, self.scale_factor)
            except Exception as e:
                messagebox.showerror("缩放错误", str(e))
                return
        
        # 2. 旋转
        if self.rotation_angle != 0:
            img = image_rotate(img, self.rotation_angle)
        
        # 3. 平移
        if self.translation != (0, 0):
            dx, dy = self.translation
            img = image_translate(img, dx, dy)
        
        self.processed_image = img
        self.update_processed_display()
        self.update_histogram()
    
    def rotate_processed(self, angle):
        """旋转处理后的图像"""
        if self.base_processed_image is not None:
            self.rotation_angle += angle
            self._apply_geometric_transforms()
    
    def transpose_processed(self):
        """转置处理后的图像"""
        if self.processed_image is not None:
            self.processed_image = image_transpose(self.processed_image)
            self.base_processed_image = self.processed_image.copy()
            self.reset_geometric_params()
            self.update_processed_display()
            self.update_histogram()
    
    def translate_processed(self):
        """平移处理后的图像"""
        if self.base_processed_image is None:
            return
            
        dx = simpledialog.askinteger("平移量", "水平平移量（向右为正）:", parent=self.master)
        if dx is None: return
        
        dy = simpledialog.askinteger("平移量", "竖直平移量（向下为正）:", parent=self.master)
        if dy is None: return
        
        self.translation = (self.translation[0] + dx, self.translation[1] + dy)
        self._apply_geometric_transforms()
    
    def zoom_processed(self, action):
        """缩放处理后的图像"""
        if self.base_processed_image is not None:
            if action == 'in':
                self.scale_factor *= 2
            else:
                self.scale_factor *= 0.5
            self._apply_geometric_transforms()
    
    def transpose_original(self):
        """转置原图"""
        if self.original_image is not None:
            self.original_image = image_transpose(self.original_image)
            self.update_original_display()
            self.update_histogram()
    
    def reset_processed_image(self):
        """重置处理后的图像"""
        if self.original_image is not None:
            self.base_processed_image = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.reset_geometric_params()
            self.gamma_slider.set(1.0)
            self.gamma_label.config(text="幂次变换 (1.00)")
            self.update_processed_display()
            self.update_histogram()

    # ============== 滤波器参数更新方法 ==============
    def update_kernel_size(self, value):
        """更新滤波器核大小"""
        self.filter_kernel_size = int(float(value))
        # 确保核大小为奇数
        self.filter_kernel_size = max(3, self.filter_kernel_size | 1)
        self.kernel_size_label.config(text=str(self.filter_kernel_size))
    
    def update_sigma_value(self, value):
        """更新高斯滤波sigma值"""
        self.gaussian_sigma = float(value)
        self.sigma_label.config(text=f"{self.gaussian_sigma:.1f}")
    
    # ============== 滤波器应用方法 ==============
    def apply_mean_filter(self):
        """应用均值滤波"""
        if self.original_image is not None:
            filtered = img_mean_filter(self.original_image, self.filter_kernel_size)
            self._update_processed_image(filtered)
    
    def apply_median_filter(self):
        """应用中值滤波"""
        if self.original_image is not None:
            filtered = img_median_filter(self.original_image, self.filter_kernel_size)
            #filtered = cupy_median_filter(self.original_image, self.filter_kernel_size)
            self._update_processed_image(filtered)
    
    def apply_gaussian_filter(self):
        """应用高斯滤波"""
        if self.original_image is not None:
            filtered = img_Gaussian_filter(
                self.original_image, 
                self.filter_kernel_size, 
                self.gaussian_sigma
            )
            self._update_processed_image(filtered)
    
    def apply_sobel_filter(self):
        """应用Sobel滤波"""
        if self.original_image is not None:
            # Sobel通常使用固定3x3核，忽略核大小参数
            filtered = img_Sobel_filter(
                self.original_image if len(self.original_image.shape) == 2 
                else cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY),
                direction='both'
            )
            self._update_processed_image(filtered)
    
    def apply_laplace_filter(self):
        """应用Laplace滤波"""
        if self.original_image is not None:
            # Laplace通常使用固定3x3核，忽略核大小参数
            filtered = img_Laplace_filter(
                self.original_image if len(self.original_image.shape) == 2 
                else cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
            )
            self._update_processed_image(filtered)

    # ============== 形态学处理方法 ==============
    def update_morph_kernel_size(self, value):
        """更新形态学核大小"""
        size = int(float(value))
        # 确保核大小为奇数
        size = max(3, size | 1)
        self.morph_kernel_label.config(text=str(size))

    def apply_dilation(self):
        """应用膨胀操作"""
        if not self._check_image_loaded():
            return
        
        # 获取当前显示图像（原图或处理后的图像）
        image = self._get_current_display_image()
        
        # 转换为灰度并二值化
        binary = self._convert_to_binary(image)
        
        # 获取参数
        kernel_size = int(self.morph_kernel_label.cget("text"))
        connectivity = int(self.connectivity_var.get())
        
        # 执行膨胀操作
        dilated = binary_dilation(binary, kernel_size, connectivity)
        
        # 更新显示
        self._update_processed_image(dilated)

    def apply_erosion(self):
        """应用腐蚀操作"""
        if not self._check_image_loaded():
            return
        
        image = self._get_current_display_image()
        binary = self._convert_to_binary(image)
        
        kernel_size = int(self.morph_kernel_label.cget("text"))
        connectivity = int(self.connectivity_var.get())
        
        eroded = binary_erosion(binary, kernel_size, connectivity)
        self._update_processed_image(eroded)

    def apply_opening(self):
        """应用开运算"""
        if not self._check_image_loaded():
            return
        
        image = self._get_current_display_image()
        binary = self._convert_to_binary(image)
        
        kernel_size = int(self.morph_kernel_label.cget("text"))
        connectivity = int(self.connectivity_var.get())
        
        opened = binary_opening(binary, kernel_size, connectivity)
        self._update_processed_image(opened)

    def apply_closing(self):
        """应用闭运算"""
        if not self._check_image_loaded():
            return
        
        image = self._get_current_display_image()
        binary = self._convert_to_binary(image)
        
        kernel_size = int(self.morph_kernel_label.cget("text"))
        connectivity = int(self.connectivity_var.get())
        
        closed = binary_closing(binary, kernel_size, connectivity)
        self._update_processed_image(closed)

    def apply_gradient(self):
        """应用形态学梯度"""
        if not self._check_image_loaded():
            return
        
        image = self._get_current_display_image()
        binary = self._convert_to_binary(image)
        
        kernel_size = int(self.morph_kernel_label.cget("text"))
        connectivity = int(self.connectivity_var.get())
        
        gradient = binary_gradient(binary, kernel_size, connectivity)
        self._update_processed_image(gradient)

    def apply_labeling(self):
        """应用连通域标记"""
        if not self._check_image_loaded():
            return
        
        image = self._get_current_display_image()
        binary = self._convert_to_binary(image)
        connectivity = int(self.connectivity_var.get())
        
        # 执行连通域标记
        labeled = connected_component_labeling(binary, connectivity)
        
        # 可视化处理
        if labeled.max() > 0:
            # 归一化到0-255
            normalized = (labeled * (255 / labeled.max())).astype(np.uint8)
            # 应用颜色映射（模拟cv2.applyColorMap）
            colored = self._apply_colormap(normalized)
            # 背景设为黑色
            colored[binary == 0] = 0
            self._update_processed_image(colored)
        else:
            self._update_processed_image(np.zeros_like(image))

    def apply_labeling(self):
        """应用连通域标记"""
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 3:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.processed_image
            
            # 二值化
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            connectivity = int(self.connectivity_var.get())
            labeled = connected_component_labeling(binary, connectivity)
            
            # 将标签图像转换为彩色以便可视化
            if labeled.max() > 0:
                # 归一化到0-255
                normalized = (labeled * (255 / labeled.max())).astype(np.uint8)
                # 应用颜色映射
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                # 背景设为黑色
                colored[binary == 0] = 0
                self._update_processed_image(colored)
            else:
                self._update_processed_image(np.zeros_like(self.processed_image))

    # ============== 辅助方法 ==============
    def _check_image_loaded(self):
        """检查是否已加载图像"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return False
        return True

    def _get_current_display_image(self):
        """获取当前显示图像（原图或处理后的图像）"""
        return self.processed_image if self.processed_image is not None else self.original_image

    def _convert_to_binary(self, image):
        """将图像转换为二值图像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 使用大津法自动阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _apply_colormap(self, gray_image):
        """模拟OpenCV的applyColorMap函数（自主实现）"""
        # 创建彩虹色映射
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            if i < 51:
                colormap[i, 0, 0] = 255
                colormap[i, 0, 1] = i * 5
                colormap[i, 0, 2] = 0
            elif i < 102:
                colormap[i, 0, 0] = 255 - (i - 51) * 5
                colormap[i, 0, 1] = 255
                colormap[i, 0, 2] = 0
            elif i < 153:
                colormap[i, 0, 0] = 0
                colormap[i, 0, 1] = 255
                colormap[i, 0, 2] = (i - 102) * 5
            elif i < 204:
                colormap[i, 0, 0] = 0
                colormap[i, 0, 1] = 255 - (i - 153) * 5
                colormap[i, 0, 2] = 255
            else:
                colormap[i, 0, 0] = (i - 204) * 5
                colormap[i, 0, 1] = 0
                colormap[i, 0, 2] = 255
        
        # 应用颜色映射
        return colormap[gray_image]

    # ============== 文件操作方法 ==============
    def open_file(self):
        """打开单个图像文件"""
        file_types = [
            ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp;*.raw;*.tif;*.tiff"), 
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(filetypes=file_types) # 弹出一个窗口让用户选择文件
        if file_path:
            directory = os.path.dirname(file_path) # 提取文件的目录部分
            self.load_images_from_directory(directory) # 从目录加载图像
            
            file_path_norm = os.path.normcase(file_path) # 规范化路径格式（统一大小写，斜杠）
            for idx, path in enumerate(self.image_paths):
                if os.path.normcase(path) == file_path_norm:
                    self.current_index = idx
                    self.show_current_image()
                    break
            else:
                messagebox.showerror("错误", "找不到文件")
    
    def open_directory(self):
        """打开图像目录"""
        directory = filedialog.askdirectory()
        if directory:
            self.load_images_from_directory(directory)
            if self.image_paths:
                self.current_index = 0
                self.show_current_image()
    
    def load_images_from_directory(self, directory):
        """从目录加载图像"""
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'raw', 'tif', 'tiff']
        self.image_paths = []
        
        for ext in extensions:
            self.image_paths += glob.glob(os.path.join(directory, f"*.{ext}")) # 创建一个含有所有directory.extensions文件目录的列表
        
        self.image_paths.sort()
        
        if not self.image_paths:
            messagebox.showinfo("提示", "该目录没有找到图片文件")
    
    def show_current_image(self):
        """显示当前图像"""
        self.reset_geometric_params()
        
        if 0 <= self.current_index < len(self.image_paths):
            file_path = self.image_paths[self.current_index]
            
            try:
                if file_path.lower().endswith('.raw'):
                    img = read_raw_image(file_path)
                else:
                    pil_img = Image.open(file_path)
                    print("图像模式:", pil_img.mode)  # 预期输出 "L"、"RGB" 等
                    img = np.array(pil_img)
                
                self.original_image = img
                self.base_processed_image = img.copy()
                self.processed_image = img.copy()
                self.gamma_slider.set(1.0)
                self.gamma_label.config(text="幂次变换 (1.00)")
                
                self.update_original_display()
                self.update_processed_display()
                self.update_histogram()
                self.update_info_label()
            except Exception as e:
                messagebox.showerror("错误", f"无法读取文件：{str(e)}")
    
    def show_previous(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def show_next(self):
        """显示下一张图像"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def save_original_image(self):
        """保存原图"""
        if self.original_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
            
        try:
            original_dir = os.path.dirname(self.image_paths[self.current_index])
            save_dir = os.path.join(original_dir, "保存结果")
            os.makedirs(save_dir, exist_ok=True)
            
            original_name = os.path.basename(self.image_paths[self.current_index])
            
            if len(self.original_image.shape) == 2:
                save_path = os.path.join(save_dir, original_name)
                self._save_raw_image(self.original_image, save_path)
            else:
                base_name = os.path.splitext(original_name)[0]
                save_path = os.path.join(save_dir, f"{base_name}.png")
                self._save_png_image(self.original_image, save_path)
                
            messagebox.showinfo("保存成功", f"原图已保存到：\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"错误原因：{str(e)}")
    
    def save_processed_image(self):
        """保存处理后的图像"""
        if self.processed_image is None:
            messagebox.showwarning("提示", "没有处理后的图像")
            return
            
        try:
            original_dir = os.path.dirname(self.image_paths[self.current_index])
            save_dir = os.path.join(original_dir, "保存结果")
            os.makedirs(save_dir, exist_ok=True)
            
            original_name = os.path.basename(self.image_paths[self.current_index])
            
            if len(self.processed_image.shape) == 2:
                save_path = os.path.join(save_dir, "processed_" + original_name)
                self._save_raw_image(self.processed_image, save_path)
            else:
                base_name = os.path.splitext(original_name)[0]
                save_path = os.path.join(save_dir, f"processed_{base_name}.png")
                self._save_png_image(self.processed_image, save_path)
                
            messagebox.showinfo("保存成功", f"处理后图像已保存到：\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"错误原因：{str(e)}")
    
    def _save_raw_image(self, image, save_path):
        """保存RAW格式图像(内部方法)"""
        if len(image.shape) != 2:
            raise ValueError("只能保存灰度图为RAW格式")
            
        with open(save_path, 'wb') as f:
            height, width = image.shape
            f.write(struct.pack('II', width, height))
            f.write(image.tobytes())
    
    def _save_png_image(self, image, save_path):
        """保存PNG格式图像(内部方法)"""
        pil_image = create_pil_image(image)
        pil_image.save(save_path)

if __name__ == "__main__":
    root = tk.Tk()
    # warmup_gpu()
    app = ImageViewerApp(root)
    root.mainloop()
