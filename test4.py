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

# 保留实验三自定义的 mybincount ,展示如何手动实现np.bincount功能。实际上我手搓的比np慢很多
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

#############################################
# 图形用户界面模块（GUI）
#############################################

class ImageViewerApp:
    """
    图像直方图分析器 GUI 应用程序类。
    主要职责：
      - 提供图像加载、浏览、处理（反色、对数变换、直方图均衡、伽马变换）和保存功能；
      - 展示原图与处理后图像及其直方图，便于对比分析。
    """
    def __init__(self, master):
        self.master = master
        self.image_paths = []
        self.current_index = -1
        self.original_image = None   # 保存原始图像（NumPy数组）
        self.processed_image = None  # 保存处理后图像（NumPy数组）
        
        self.master.title("数字图像处理实验")
        self.master.geometry("1200x900")
        self.setup_ui()
    
    def setup_ui(self):
        """初始化并布局所有界面组件。"""
        # 顶部按钮区：文件操作和保存原图
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
        
        # 底部按钮区：图像处理操作（不含伽马变换控件）和保存处理后图像
        bottom_button_frame = ttk.Frame(self.master)
        bottom_button_frame.pack(pady=10)
        processing_buttons = [
            ("反色", self.apply_invert),
            ("对数变换", self.apply_log_transform),
            ("直方图均衡", self.apply_histogram_equalization),
            ("保存处理后的图像", self.save_processed_image)
        ]
        for text, command in processing_buttons:
            ttk.Button(bottom_button_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
        
        # 图像显示区：分左右两部分显示原图和处理后的图像
        main_image_frame = ttk.Frame(self.master)
        main_image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：原图显示区域
        self.left_frame = ttk.Frame(main_image_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.image_canvas_left = tk.Canvas(self.left_frame, bg='#e0e0e0')
        self.image_canvas_left.pack(fill=tk.BOTH, expand=True)
        
        # 右侧：处理后图像显示区域及伽马变换控件
        self.right_frame = ttk.Frame(main_image_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 伽马变换控件：标签和滑动条，放置于右侧上方
        self.gamma_frame = ttk.Frame(self.right_frame)
        self.gamma_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        # 初始标签增加实时伽马值显示，初始值为1.00
        self.gamma_label = ttk.Label(self.gamma_frame, text="幂次变换 (1.00)")
        self.gamma_label.pack(anchor='w')
        self.gamma_slider = ttk.Scale(self.gamma_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, command=self.update_gamma_preview)
        self.gamma_slider.set(1.0)
        self.gamma_slider.pack(fill=tk.X)
        
        # 处理后图像的画布，位于伽马控件下方
        self.image_canvas_right = tk.Canvas(self.right_frame, bg='#e0e0e0')
        self.image_canvas_right.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 图像信息标签
        self.info_label = ttk.Label(self.master, text="图片信息：未加载")
        self.info_label.pack(pady=5)
        
        # 直方图显示区：利用 Matplotlib 展示原图与处理后图像的直方图
        self.histogram_frame = ttk.Frame(self.master)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.figure = Figure(figsize=(10, 3))
        self.hist_canvas = FigureCanvasTkAgg(self.figure, self.histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def open_file(self):
        """通过文件对话框打开单个图像文件。"""
        file_types = [
            ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp;*.raw"),
            ("所有文件", "*.*")
        ]
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
        """通过目录对话框打开图像文件夹。"""
        directory = filedialog.askdirectory()
        if directory:
            self.load_images_from_directory(directory)
            if self.image_paths:
                self.current_index = 0
                self.show_current_image()
    
    def load_images_from_directory(self, directory):
        """
        从指定目录加载所有支持的图像文件（jpg, jpeg, png, bmp, raw）。
        文件路径将排序后存入 self.image_paths。
        """
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'raw']
        self.image_paths = []
        for ext in extensions:
            self.image_paths += glob.glob(os.path.join(directory, f"*.{ext}"))
        self.image_paths.sort()
        if not self.image_paths:
            messagebox.showinfo("提示", "该目录没有找到图片文件")
    
    def show_current_image(self):
        """加载当前图像文件，并在左右画布中显示原图与处理后的图像及直方图。"""
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return
        file_path = self.image_paths[self.current_index]
        try:
            if file_path.lower().endswith('.raw'):
                img = read_raw_image(file_path)
            else:
                pil_image = Image.open(file_path)
                img = np.array(pil_image)
            self.original_image = img      # 保存原图
            self.processed_image = img.copy()  # 初始化处理图与原图一致
            self.gamma_slider.set(1.0)  # 重置伽马滑动条
            # 重置伽马标签为初始值
            self.gamma_label.config(text="幂次变换 (1.00)")
            self.update_original_display()
            self.update_processed_display()
            self.update_histogram()
            self.update_info_label()
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件：{str(e)}")
    
    def update_original_display(self):
        """在左侧画布中显示原图像。"""
        if self.original_image is None:
            return
        pil_image = create_pil_image(self.original_image)
        canvas = self.image_canvas_left
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 400
            canvas_height = 400
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_image)
        canvas.image = tk_image  # 防止图像被回收
    
    def update_processed_display(self):
        """在右侧画布中显示处理后的图像。"""
        if self.processed_image is None:
            return
        pil_image = create_pil_image(self.processed_image)
        canvas = self.image_canvas_right
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas_width = 400
            canvas_height = 400
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, image=tk_image)
        canvas.image = tk_image
    
    def update_info_label(self):
        """更新图像信息标签，显示尺寸、通道数及当前图片索引。"""
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
        """更新并显示原图和处理后图像的直方图（使用 Matplotlib 绘制）。"""
        self.figure.clear()
        # 左侧直方图：原图
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
        # 右侧直方图：处理后图像
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
        """显示上一张图像。"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def show_next(self):
        """显示下一张图像。"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def save_original_image(self):
        """将原始图像保存到“保存结果”目录中。"""
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
        """将处理后的图像保存到“保存结果”目录中。"""
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
        """
        以 RAW 格式保存灰度图像。
        RAW 格式包括 8 字节头部（宽度和高度）以及图像数据。
        """
        if len(image.shape) != 2:
            raise ValueError("只能保存灰度图为RAW格式")
        with open(save_path, 'wb') as f:
            height, width = image.shape
            f.write(struct.pack('II', width, height))
            f.write(image.tobytes())
    
    def _save_png_image(self, image, save_path):
        """
        以 PNG 格式保存图像（彩色或灰度）。
        利用 PIL 库实现图像保存。
        """
        pil_image = create_pil_image(image)
        pil_image.save(save_path)
    
    # ================ 图像处理功能接口（调用核心图像处理函数） ================= 
    def apply_invert(self):
        """调用反色处理函数，并更新显示。"""
        if self.original_image is not None:
            self.processed_image = invert_image(self.original_image)
            self.update_processed_display()
            self.update_histogram()
    
    def apply_log_transform(self):
        """调用对数变换增强函数，并更新显示。"""
        if self.original_image is None:
            return
        c = simpledialog.askfloat("对数变换", "请输入增强系数(范围1-5)：", 
                                  parent=self.master, minvalue=1.0, maxvalue=5.0)
        if c is None:
            return
        self.processed_image = log_transform_image(self.original_image, c)
        self.update_processed_display()
        self.update_histogram()
    
    def apply_histogram_equalization(self):
        """调用直方图均衡化函数，并更新显示。"""
        if self.original_image is None:
            return
        self.processed_image = equalize_histogram(self.original_image)
        self.update_processed_display()
        self.update_histogram()
    
    def update_gamma_preview(self, value):
        """
        根据滑动条数值实时预览伽马变换效果，更新处理后图像。
        同时更新显示在标签上的当前伽马值，保留两位小数。
        """
        if self.original_image is None:
            return
        gamma = float(value)
        # 更新标签文字，显示实时伽马值（保留两位小数）
        self.gamma_label.config(text=f"幂次变换 ({gamma:.2f})")
        self.processed_image = gamma_transform(self.original_image, gamma)
        self.update_processed_display()
        self.update_histogram()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()
