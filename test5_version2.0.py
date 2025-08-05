import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import os
import glob
import struct
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib

# 配置 Matplotlib 字体，确保中文显示正常
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

#############################################
# 图像处理核心函数模块
#############################################

def mybincount(arr, minlength=None):
    """
    自定义实现的数组计数函数，用于统计数组中每个整数出现的次数。
    注意：
      - 仅适用于非负整数数组；
      - 为了优化运行速度，后面没有用到这个函数，而是使用了np.bincount。
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("mybincount() 要求输入数组的数据类型为整数")
    if np.any(arr < 0):
        raise ValueError("mybincount() 不支持负数")
    if arr.size == 0:
        max_val = 0
    else:
        max_val = int(arr.max())
    if minlength is None:
        length = max_val + 1
    else:
        length = max(max_val + 1, int(minlength))
    counts = np.zeros(length, dtype=int)
    np.add.at(counts, arr, 1)
    return counts

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

def create_pil_image(numpy_array):
    """
    将 NumPy 数组转换为 PIL Image 对象。
    根据数组维度判断是彩色图像还是灰度图像。
    """
    if len(numpy_array.shape) == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), 'RGB')
    else:
        return Image.fromarray(numpy_array.astype(np.uint8), 'L')

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
    hist = np.bincount(channel.ravel(), minlength=256)
    cdf = hist.cumsum()
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

'''
这个是PIL的放大，测试用的
def image_zoom_in(image, zoom_factor=2):
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    
    # 使用PIL的resize实现高质量放大
    pil_img = create_pil_image(image)
    return np.array(pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS))
'''
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

#############################################
# 图形用户界面模块（改进版）
#############################################

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.image_paths = []
        self.current_index = -1
        self.original_image = None       # 原始图像数据
        self.base_processed_image = None # 基础处理结果（如均衡化、反色等处理后的图像）
        self.processed_image = None      # 最终显示图像（基础处理+几何变换）

        # 几何变换参数
        self.rotation_angle = 0         # 累计旋转角度
        self.scale_factor = 1.0         # 当前缩放比例
        self.translation = (0, 0)       # 平移量 (dx, dy)

        self.master.title("数字图像处理实验(改进版)")
        self.master.geometry("1400x950")
        self.setup_ui()

    def setup_ui(self):
        """初始化并布局所有界面组件"""
        """顶层基本操作按钮"""
        top_button_frame = ttk.Frame(self.master)
        top_button_frame.pack(pady=10)
        top_buttons = [
            ("打开文件", self.open_file),
            ("打开目录", self.open_directory),
            ("上一张", self.show_previous),
            ("下一张", self.show_next),
            ("保存原图", self.save_original_image)
        ]
        for text, command in top_buttons:
            ttk.Button(top_button_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)

        """下层框架，左侧为基本图像变换按钮，右侧为伽马变换条"""
        bottom_main_frame = ttk.Frame(self.master)
        bottom_main_frame.pack(pady=10, fill=tk.X)

        processing_buttons_frame = ttk.Frame(bottom_main_frame)
        processing_buttons_frame.pack(side=tk.LEFT, padx=20)
        processing_buttons = [
            ("反色", self.apply_invert),
            ("对数变换", self.apply_log_transform),
            ("直方图均衡", self.apply_histogram_equalization),
            ("保存处理后的图像", self.save_processed_image)
        ]
        for text, command in processing_buttons:
            ttk.Button(processing_buttons_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)

        #伽马控件部分
        gamma_control_frame = ttk.Frame(bottom_main_frame)
        gamma_control_frame.pack(side=tk.RIGHT, padx=20)
        self.gamma_label = ttk.Label(gamma_control_frame, text="幂次变换 (1.00)")
        self.gamma_label.pack(anchor='w')
        self.gamma_slider = ttk.Scale(gamma_control_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=500, command=self.update_gamma_preview)
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(fill=tk.X)

        main_image_frame = ttk.Frame(self.master)
        main_image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_image_frame.columnconfigure(0, weight=1)
        main_image_frame.columnconfigure(1, weight=1)
        main_image_frame.rowconfigure(0, weight=1)

        # 左侧原图画布与旋转按钮
        self.left_frame = ttk.Frame(main_image_frame)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        original_btn_frame = ttk.Frame(self.left_frame)
        original_btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(original_btn_frame, text="原图逆时针旋转90", command=self.transpose_original).pack(padx=2)
        self.image_canvas_left = tk.Canvas(self.left_frame, bg='#e0e0e0')
        self.image_canvas_left.pack(fill=tk.BOTH, expand=True)

        # 右侧修改后图像与旋转缩放按钮
        self.right_frame = ttk.Frame(main_image_frame)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        processed_btn_frame = ttk.Frame(self.right_frame)
        processed_btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(processed_btn_frame, text="逆时针旋转90", command=self.transpose_processed).pack(side=tk.LEFT, padx=2)
        ttk.Button(processed_btn_frame, text="逆时针旋转10", command=lambda: self.rotate_processed(10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(processed_btn_frame, text="平移", command=self.translate_processed).pack(side=tk.LEFT, padx=2)
        ttk.Button(processed_btn_frame, text="放大两倍", command=lambda: self.zoom_processed('in')).pack(side=tk.LEFT, padx=2)
        ttk.Button(processed_btn_frame, text="缩小1/2", command=lambda: self.zoom_processed('out')).pack(side=tk.LEFT, padx=2)
        ttk.Button(processed_btn_frame, text="恢复", command=self.reset_processed_image).pack(side=tk.LEFT, padx=2)
        
        #要显示缩放，不能自适应，而是要显示原大小、加入滚动条
        #self.image_canvas_right = tk.Canvas(self.right_frame, bg='#e0e0e0')
        #self.image_canvas_right.pack(fill=tk.BOTH, expand=True)
        # 替换原来的右侧画布布局代码
        # 在setup_ui()中修改右侧画布初始化：
        right_container = ttk.Frame(self.right_frame)
        right_container.pack(fill=tk.BOTH, expand=True)

        self.image_canvas_right = tk.Canvas(right_container, bg='#e0e0e0')
        vsb = ttk.Scrollbar(right_container, orient="vertical", command=self.image_canvas_right.yview)
        hsb = ttk.Scrollbar(right_container, orient="horizontal", command=self.image_canvas_right.xview)

        self.image_canvas_right.configure(
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            highlightthickness=0  # 去除边框线
        )
        # 网格布局
        self.image_canvas_right.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)


        #直方图部分
        self.info_label = ttk.Label(self.master, text="图片信息：未加载")
        self.info_label.pack(pady=5)
        self.histogram_frame = ttk.Frame(self.master)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.figure = Figure(figsize=(10, 2))
        self.hist_canvas = FigureCanvasTkAgg(self.figure, self.histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_geometric_params(self):
        """重置所有几何变换参数"""
        self.rotation_angle = 0
        self.scale_factor = 1.0
        self.translation = (0, 0)


    def _apply_geometric_transforms(self):
        """
        应用所有几何变换到基础处理图像上
        顺序: 缩放 -> 旋转 -> 平移
        """
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

    # ============== 图像加载与显示相关方法 ==============
    def open_file(self):
        file_types = [("图片文件", "*.jpg;*.jpeg;*.png;*.bmp;*.raw;*.tif;*.tiff"), ("所有文件", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            directory = os.path.dirname(file_path)
            self.load_images_from_directory(directory)
            file_path_norm = os.path.normcase(file_path)
            matched_index = None
            for idx, path in enumerate(self.image_paths):
                if os.path.normcase(path) == file_path_norm:
                    matched_index = idx
                    break
            if matched_index is not None:
                self.current_index = matched_index
                self.show_current_image()
            else:
                messagebox.showerror("错误", "找不到文件")

    def open_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.load_images_from_directory(directory)
            if self.image_paths:
                self.current_index = 0
                self.show_current_image()

    def load_images_from_directory(self, directory):
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'raw', 'tif', 'tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths += glob.glob(os.path.join(directory, f"*.{ext}"))
        self.image_paths.sort()
        if not self.image_paths:
            messagebox.showinfo("提示", "该目录没有找到图片文件")

    def show_current_image(self):
        """加载当前图像文件，并重置所有参数"""
        self.reset_geometric_params()
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return
        file_path = self.image_paths[self.current_index]
        try:
            if file_path.lower().endswith('.raw'):
                img = read_raw_image(file_path)
            else:
                pil_image = Image.open(file_path)
                img = np.array(pil_image)
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

    def update_original_display(self):
        if self.original_image is None:
            return
        pil_image = create_pil_image(self.original_image)
        canvas = self.image_canvas_left
        canvas_width = canvas.winfo_width() or 400
        # canvas_height = canvas.winfo_height() or 400
        canvas_height = 400 #Tkinter的自适应有bug，前一张是大图，后一张是小图时，原图显示不全。所以固定400。
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_image)
        canvas.image = tk_image
    '''

    #==============================图像更新===============================
    原本将图像按画布大小缩放显示的display
    def update_processed_display(self):
        if self.processed_image is None:
            return
        pil_image = create_pil_image(self.processed_image)
        canvas = self.image_canvas_right
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 400
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_image)
        canvas.image = tk_image
    '''
    def update_processed_display(self):
        if self.processed_image is None:
            return
            
        # 将处理后图像转换为PIL对象
        pil_image = create_pil_image(self.processed_image)
        img_width, img_height = pil_image.size
        
        # 获取原图尺寸作为参考（如果原图存在）
        if self.original_image is not None:
            orig_pil = create_pil_image(self.original_image)
            orig_width, orig_height = orig_pil.size
        else:
            orig_width, orig_height = img_width, img_height
        
        # 创建足够大的画布区域（至少显示完整原图尺寸）
        canvas_width = max(img_width, orig_width)
        canvas_height = max(img_height, orig_height)
        
        # 创建黑色背景图像
        background = Image.new("RGB", (canvas_width, canvas_height), "black")
        
        # 计算居中位置
        paste_x = (canvas_width - img_width) // 2
        paste_y = (canvas_height - img_height) // 2
        
        # 将处理后的图像粘贴到居中位置
        background.paste(pil_image, (paste_x, paste_y))
        
        # 转换为Tkinter可显示格式
        tk_image = ImageTk.PhotoImage(background)
        
        # 更新画布
        canvas = self.image_canvas_right
        canvas.delete("all")
        
        # 设置画布滚动区域
        canvas.config(
            scrollregion=(0, 0, canvas_width, canvas_height),
            width=min(canvas_width, 800),  # 限制最大初始显示宽度
            height=min(canvas_height, 600)  # 限制最大初始显示高度
        )
        
        # 创建图像并居中显示
        canvas.create_image(
            canvas_width//2, 
            canvas_height//2, 
            anchor=tk.CENTER, 
            image=tk_image
        )
        canvas.image = tk_image  # 防止图像被垃圾回收
        
        # 自动滚动到中心位置
        canvas.xview_moveto(0.5 - (400/canvas_width))
        canvas.yview_moveto(0.5 - (300/canvas_height))



    def update_info_label(self):
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
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        if self.original_image is not None:
            if len(self.original_image.shape) == 2:
                gray_hist = calculate_gray_histogram(self.original_image)
                ax1.bar(range(256), gray_hist, width=1.0, color='gray')
            else:
                r_hist, g_hist, b_hist = calculate_color_histogram(self.original_image)
                ax1.plot(range(256), r_hist, color='red')
                ax1.plot(range(256), g_hist, color='green')
                ax1.plot(range(256), b_hist, color='blue')
            ax1.set_title("原图直方图")
        ax2 = self.figure.add_subplot(122)
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:
                gray_hist = calculate_gray_histogram(self.processed_image)
                ax2.bar(range(256), gray_hist, width=1.0, color='gray')
            else:
                r_hist, g_hist, b_hist = calculate_color_histogram(self.processed_image)
                ax2.plot(range(256), r_hist, color='red')
                ax2.plot(range(256), g_hist, color='green')
                ax2.plot(range(256), b_hist, color='blue')
            ax2.set_title("处理后直方图")
        self.figure.tight_layout()
        self.hist_canvas.draw()

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()

    def save_original_image(self):
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
        if len(image.shape) != 2:
            raise ValueError("只能保存灰度图为RAW格式")
        with open(save_path, 'wb') as f:
            height, width = image.shape
            f.write(struct.pack('II', width, height))
            f.write(image.tobytes())

    def _save_png_image(self, image, save_path):
        pil_image = create_pil_image(image)
        pil_image.save(save_path)

    # ============== 图像处理功能接口（调用核心函数） ==============
    def apply_invert(self):
        """反色处理（基于原始图像）"""
        if self.original_image is not None:
            self.base_processed_image = invert_image(self.original_image)
            self._apply_geometric_transforms()

    def apply_log_transform(self):
        """对数变换（基于原始图像）"""
        if self.original_image is None:
            return
        c = simpledialog.askfloat("对数变换", "请输入增强系数(范围1-5)：", parent=self.master, minvalue=1.0, maxvalue=5.0)
        if c is None:
            return
        self.base_processed_image = log_transform_image(self.original_image, c)
        self._apply_geometric_transforms()

    def apply_histogram_equalization(self):
        """直方图均衡化（基于原始图像）"""
        if self.original_image is None:
            return
        self.base_processed_image = equalize_histogram(self.original_image)
        self._apply_geometric_transforms()

    def update_gamma_preview(self, value):
        """伽马变换（基于原始图像）"""
        if self.original_image is None:
            return
        gamma = float(value)
        self.gamma_label.config(text=f"幂次变换 ({gamma:.2f})")
        self.base_processed_image = gamma_transform(self.original_image, gamma)
        self._apply_geometric_transforms()

    # ============== 几何变换操作 ==============
    def rotate_processed(self, angle):
        """旋转处理后图像（累计角度）"""
        if self.base_processed_image is None:
            return
        self.rotation_angle += angle
        self._apply_geometric_transforms()

    def transpose_processed(self):
        """逆时针旋转90°（无损操作，直接更新base_processed_image）"""
        if self.processed_image is None:
            return
        
        # 直接对当前处理后的图像进行转置
        self.processed_image = image_transpose(self.processed_image)
        
        # 关键点：同步更新base_processed_image
        self.base_processed_image = self.processed_image.copy()
        
        # 重置其他几何变换参数（因为已直接修改了基础图像）
        self.rotation_angle = 0
        self.scale_factor = 1.0
        self.translation = (0, 0)
        
        self.update_processed_display()
        self.update_histogram()

    def translate_processed(self):
        """平移图像"""
        if self.base_processed_image is None:
            return
        dx = simpledialog.askinteger("平移量", "水平平移量（向右为正）:", parent=self.master)
        if dx is None: return
        dy = simpledialog.askinteger("平移量", "竖直平移量（向下为正）:", parent=self.master)
        if dy is None: return
        self.translation = (self.translation[0] + dx, self.translation[1] + dy)
        self._apply_geometric_transforms()

    def zoom_processed(self, action):
        """缩放图像"""
        if self.base_processed_image is None:
            return
        if action == 'in':
            self.scale_factor *= 2
        else:
            self.scale_factor *= 0.5
        self._apply_geometric_transforms()

    def transpose_original(self):
        """原图逆时针旋转90度"""
        if self.original_image is not None:
            self.original_image = image_transpose(self.original_image)
            self.update_original_display()
            self.update_histogram()

    def reset_processed_image(self):
        """重置所有处理，恢复为原图"""
        if self.original_image is None:
            return
        
        # 重置所有处理状态
        self.base_processed_image = self.original_image.copy()
        self.processed_image = self.original_image.copy()
        self.reset_geometric_params()  # 重置旋转/缩放/平移参数
        self.gamma_slider.set(1.0)    # 重置伽马值
        self.gamma_label.config(text="幂次变换 (1.00)")
        
        # 更新显示
        self.update_processed_display()
        self.update_histogram()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
