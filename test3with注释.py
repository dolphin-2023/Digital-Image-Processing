"""
数字图像处理实验 - 直方图分析
作者：Harry Zhao
日期：2025-03-11

本程序使用Python的图形界面和图像处理库，实现了对图片的加载、显示、
直方图计算与显示，以及图片保存等功能。代码中详细解释了每一步骤，便于初学者学习。
"""

# ========== 导入所需的库 ==========
# tkinter：Python内置的GUI库，用于创建图形界面窗口和按钮
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# PIL（Python Imaging Library）: 用于图像处理，如打开、转换、保存图片
from PIL import Image, ImageTk

# numpy：数值计算库，用于处理图像数据（例如数组操作）
import numpy as np

# os和glob：用于处理文件和目录操作
import os
import glob

# struct：用于解析二进制数据，本程序用于解析RAW图片的文件头
import struct

# matplotlib：用于绘图，这里用来显示图片的直方图
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 配置matplotlib，使其能够显示中文（例如直方图标题中的中文）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体，如“SimHei”或“Microsoft YaHei”
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

# ========== 图像处理函数 ==========

def read_raw_image(file_path):
    """
    读取自定义的RAW格式图片。
    
    文件结构：
      - 前8个字节：保存图片宽度和高度（每个占4字节，无符号整数）
      - 后续字节：保存图片的像素数据（灰度值，范围0-255）
    
    参数:
      file_path: RAW文件的路径（字符串）
    
    返回:
      一个numpy数组，形状为 (height, width)
      
    如果文件头不足8个字节，则抛出异常，提示文件格式错误。
    """
    with open(file_path, 'rb') as file:
        # 读取文件前8个字节，获取图片宽度和高度
        header = file.read(8)
        if len(header) != 8:
            # 如果读取的字节数不是8，说明文件格式不正确
            raise ValueError(f"文件头字节数错误：期望8字节，实际读取到{len(header)}字节")
        
        # 使用struct.unpack解析8字节数据，'II'表示两个无符号整数
        width, height = struct.unpack('II', header)
        
        # 读取剩下的所有字节，作为图片像素数据
        pixels = file.read()
        
        # 使用numpy将像素数据转换成数组，并按(height, width)重塑数组形状
        return np.frombuffer(pixels, dtype=np.uint8).reshape((height, width))


def create_pil_image(numpy_array):
    """
    将numpy数组转换为PIL库的图片对象。
    
    自动判断数组是灰度图还是彩色图：
      - 灰度图：二维数组，模式为 'L'
      - 彩色图：三维数组，模式为 'RGB'
    
    参数:
      numpy_array: 保存图像数据的numpy数组
    
    返回:
      PIL.Image对象，可用于显示或保存图片
    """
    # 如果数组有三个维度，认为是彩色图（高, 宽, 3通道）
    if len(numpy_array.shape) == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), 'RGB')
    else:  # 否则为灰度图（二维数组）
        return Image.fromarray(numpy_array.astype(np.uint8), 'L')

def mybincount(arr, minlength = None):
    '''
    不给用numpy.bincount，那我手搓一个bincount
    首先明确一下np中bincount这个函数的功能：
    输入一个numpy整型全正数组、以及数组最小长度
    统计从零到最大整数出现的数目

    比如
    # arr = np.array([1, 2, 1, 3, 0, 2, 1])
    # counts = np.bincount(arr)
    # print(counts)
    # 输出: [1 3 2 1]

    实现方法很简单，遍历输入数组，将每一个值作为输出数组的位置，输出数组对应位置值+1
    '''
    #首先考虑健壮性。1)非整型报错转为整型 2)负数报错 3)空数组返回全0
    if not np.issubdtype(arr.dtype,np.integer):
        raise ValueError("mybincount() 要求输入数组的数据类型为整数")
    if np.any(arr < 0):
        raise ValueError("mybincount() 不支持负数")
    if arr.size == 0:
        max_val = 0
    else:
        max_val = int(arr.max())
    
    #根据minlength确定输出数组长度。默认为最大值+1，若有规定minlength则在最大值+1与minlength中取大的
    if minlength == None:
        length = max_val + 1 
    else:
        length = max(max_val + 1, int(minlength))
    
    #初始化输出数组(整型)
    counts = np.zeros(length, dtype = int)

    # 使用 np.add.at 进行向量化的索引累加
    np.add.at(counts, arr, 1)

    return counts

def calculate_gray_histogram(image_data):
    """
    计算灰度图像的直方图。
    
    直方图表示0到255每个灰度值出现的概率。
    
    参数:
      image_data: 灰度图像的numpy数组
    
    返回:
      一个长度为256的数组，每个元素表示相应灰度值的出现概率
    """
    # 将二维数组展开成一维，使用np.bincount统计每个灰度值出现的次数
    counts = mybincount(image_data.ravel(), minlength=256)
    # 将每个灰度值的计数除以总像素数，得到出现概率
    return counts / image_data.size


def calculate_color_histogram(image_data):
    """
    计算彩色图像的直方图。
    
    分别计算红、绿、蓝三个颜色通道的直方图，并返回每个通道的灰度值概率。
    
    参数:
      image_data: 彩色图像的numpy数组，形状为 (height, width, 3)
    
    返回:
      三个长度为256的数组，分别对应红、绿、蓝三个通道的直方图
    """
    # 分离红、绿、蓝三个通道
    red_channel = image_data[:, :, 0]
    green_channel = image_data[:, :, 1]
    blue_channel = image_data[:, :, 2]
    
    # 分别计算每个通道的直方图（统计每个灰度值的出现次数，再转换为概率）
    red_hist = mybincount(red_channel.ravel(), minlength=256) / red_channel.size
    green_hist = mybincount(green_channel.ravel(), minlength=256) / green_channel.size
    blue_hist = mybincount(blue_channel.ravel(), minlength=256) / blue_channel.size
    
    return red_hist, green_hist, blue_hist


# ========== 主程序界面类 ==========

class ImageViewerApp:
    """
    图像直方图分析器的图形界面类。
    
    提供功能：
      - 打开单个图片或整个目录
      - 显示图片及其直方图（灰度图或彩色图）
      - 支持上一张、下一张图片的浏览
      - 保存当前图片
    """
    def __init__(self, master):
        """
        初始化界面和相关变量。
        
        参数:
          master: tkinter的根窗口
        """
        self.master = master
        self.image_paths = []    # 用于存储图片文件路径的列表
        self.current_index = -1  # 当前显示的图片在列表中的索引
        self.current_image = None  # 当前加载的图片数据（numpy数组）
        
        # 设置窗口标题和初始大小
        self.master.title("实验三直方图")
        self.master.geometry("1000x800")
        
        # 创建界面上所有的组件（按钮、标签、画布等）
        self.setup_ui()
    
    def setup_ui(self):
        """
        创建所有界面组件，并安排它们在窗口中的布局。
        """
        # 创建顶部按钮区域
        button_frame = ttk.Frame(self.master)
        button_frame.pack(pady=10)
        
        # 定义按钮名称和对应的功能方法
        buttons = [
            ("打开文件", self.open_file),
            ("打开目录", self.open_directory),
            ("上一张", self.show_previous),
            ("下一张", self.show_next),
            ("保存图片", self.save_image)
        ]
        
        # 遍历按钮列表，依次创建按钮控件
        for text, command in buttons:
            ttk.Button(button_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
        
        # 创建显示图片信息的标签（初始显示为“未加载”）
        self.info_label = ttk.Label(self.master, text="图片信息：未加载")
        self.info_label.pack(pady=5)
        
        # 创建用于显示图片的Canvas控件，设置背景色，并允许自动扩展大小
        self.image_canvas = tk.Canvas(self.master, bg='#e0e0e0')#浅灰色
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_on_canvas = None  # 用于记录Canvas上显示的图片对象
        
        # 创建直方图显示区域，使用matplotlib绘图
        self.histogram_frame = ttk.Frame(self.master)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建一个matplotlib的Figure对象，并嵌入到tkinter的界面中
        self.figure = Figure(figsize=(8, 3))
        self.hist_canvas = FigureCanvasTkAgg(self.figure, self.histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # ========== 核心功能方法 ==========
    
    def open_file(self):
        """
        打开单个图片文件。选择文件后：
          1. 自动加载该文件所在目录下的所有图片到file_path；
          2. 定位所选文件在图片列表中的位置，并显示该图片。
        """
        # 定义可选的文件类型
        file_types = [
            ("图片文件", "*.jpg;*.jpeg;*.png;*.bmp;*.raw"),
            ("所有文件", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            # 获取文件所在的目录，并加载目录下所有支持的图片
            directory = os.path.dirname(file_path)
            self.load_images_from_directory(directory)
            
            # 使用不区分大小写的方式，确定选中文件在图片列表中的索引
            file_path_norm = os.path.normcase(file_path)#标准化处理
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
        """
        打开一个目录，调用load_images_from_directory加载目录下所有支持的图片，并显示第一张图片。
        """
        directory = filedialog.askdirectory()
        if directory:
            self.load_images_from_directory(directory)
            if self.image_paths:
                self.current_index = 0#表示列表中的第一张图片。index是索引
                self.show_current_image()
    
    def load_images_from_directory(self, directory):
        """
        加载指定目录下所有支持的图片文件，并将它们的路径存储在列表image_paths中。
        
        参数:
          directory: 目录的路径（字符串）
        """
        # 定义支持的图片文件扩展名
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'raw']
        self.image_paths = []
        
        # 遍历所有扩展名，使用glob模块查找匹配的文件
        for ext in extensions:
            self.image_paths += glob.glob(os.path.join(directory, f"*.{ext}"))
        
        # 将图片路径按照文件名排序，便于浏览
        self.image_paths.sort()
        
        # 如果目录中没有找到任何图片，给出提示信息
        if not self.image_paths:
            messagebox.showinfo("提示", "该目录没有找到图片文件")
    
    def show_current_image(self):
        """
        根据当前索引显示对应的图片，同时更新图片显示、直方图以及图片信息标签。
        """
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return
        
        file_path = self.image_paths[self.current_index]
        try:
            # 如果文件扩展名为.raw，则调用read_raw_image函数读取
            if file_path.lower().endswith('.raw'):
                self.current_image = read_raw_image(file_path)
            else:
                # 使用PIL库打开图片，并转换为numpy数组
                pil_image = Image.open(file_path)
                self.current_image = np.array(pil_image)
            
            # 更新图片在Canvas中的显示、直方图和信息标签
            self.update_image_display()
            self.update_histogram()
            self.update_info_label()
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件：{str(e)}")
    
    def update_image_display(self):
        """
        更新Canvas中显示的图片。
        
        功能：
          1. 将numpy数组转换为PIL图片；
          2. 根据Canvas的当前尺寸对图片进行等比例缩放；
          3. 在Canvas中居中显示图片。
        """
        # 将numpy数组转换为PIL图片对象
        pil_image = create_pil_image(self.current_image)
        
        # 获取Canvas当前的宽度和高度
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # 计算图片缩放比例，保证图片不变形（保持原始宽高比）
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        
        # 对图片进行缩放处理
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        
        # 清除Canvas中之前的图片，并在中间位置绘制新图片
        self.image_canvas.delete("all")
        self.image_on_canvas = self.image_canvas.create_image(
            canvas_width // 2,   # X坐标居中
            canvas_height // 2,  # Y坐标居中
            image=tk_image       # 使用处理后的图片
        )
        
        # 保存图片引用，防止图片被垃圾回收而无法显示
        self.image_canvas.tk_image = tk_image
    
    def update_info_label(self):
        """
        更新窗口中显示图片信息的标签，信息包括：
          - 图片尺寸（宽×高）
          - 图片通道数（1表示灰度图，3表示彩色图）
          - 当前图片在目录中的序号
        """
        if self.current_image is None:
            return
        
        # 判断图片数据是二维（灰度图）还是三维（彩色图）
        if len(self.current_image.shape) == 2:
            height, width = self.current_image.shape
            channels = 1
        else:
            height, width, channels = self.current_image.shape
        
        info_text = f"尺寸：{width}×{height} 像素 | 通道：{channels} | 当前：{self.current_index + 1}/{len(self.image_paths)}"
        self.info_label.config(text=info_text)
    
    def update_histogram(self):
        """
        根据当前加载的图片数据计算直方图，并在界面中显示。
        
        如果是灰度图，绘制一幅直方图；
        如果是彩色图，分别绘制红、绿、蓝三个通道的直方图。
        """
        # 清除之前绘制的图像
        self.figure.clear()
        
        # 判断图片数据的维度来区分灰度图和彩色图
        if len(self.current_image.shape) == 2:
            # 计算灰度直方图并绘制
            gray_hist = calculate_gray_histogram(self.current_image)
            ax = self.figure.add_subplot(111)
            ax.bar(range(256), gray_hist, width=1.0, color='gray')
            ax.set_title("灰度直方图")
        else:
            # 计算彩色图的红、绿、蓝直方图
            r_hist, g_hist, b_hist = calculate_color_histogram(self.current_image)
            
            # 在三个子图中分别绘制不同颜色通道的直方图
            ax = self.figure.add_subplot(131)
            ax.bar(range(256), r_hist, width=1.0, color='red')
            ax.set_title("红色通道")
            
            ax = self.figure.add_subplot(132)
            ax.bar(range(256), g_hist, width=1.0, color='green')
            ax.set_title("绿色通道")
            
            ax = self.figure.add_subplot(133)
            ax.bar(range(256), b_hist, width=1.0, color='blue')
            ax.set_title("蓝色通道")
        
        # 调整图像布局，使各部分不重叠，并刷新显示直方图
        self.figure.tight_layout()
        self.hist_canvas.draw()
    
    def show_previous(self):
        """
        显示上一张图片。
        如果当前不是第一张，则将索引减1，并调用显示图片的函数。
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def show_next(self):
        """
        显示下一张图片。
        如果当前不是最后一张，则将索引加1，并调用显示图片的函数。
        """
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def save_image(self):
        """
        保存当前显示的图片。
        
        如果是灰度图，则保存为RAW格式（保持原始数据结构）；
        如果是彩色图，则保存为PNG格式。
        保存路径为原文件所在目录下的“保存结果”子目录。
        """
        if self.current_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        
        try:
            # 获取当前图片所在的目录，并创建“保存结果”目录（如果不存在）
            original_dir = os.path.dirname(self.image_paths[self.current_index])
            save_dir = os.path.join(original_dir, "保存结果")
            os.makedirs(save_dir, exist_ok=True)
            
            # 获取原始文件名，并根据图片类型生成保存路径
            original_name = os.path.basename(self.image_paths[self.current_index])
            if len(self.current_image.shape) == 2:  # 灰度图保存为RAW格式
                save_path = os.path.join(save_dir, original_name)
                self.save_raw_image(save_path)
            else:  # 彩色图保存为PNG格式
                base_name = os.path.splitext(original_name)[0]
                save_path = os.path.join(save_dir, f"{base_name}.png")
                self.save_png_image(save_path)
            
            messagebox.showinfo("保存成功", f"图片已保存到：\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"错误原因：{str(e)}")
    
    def save_raw_image(self, save_path):
        """
        保存灰度图为RAW格式。
        
        RAW格式要求图像数据为二维数组，
        首先写入宽度和高度（每个4字节），然后写入像素数据。
        
        参数:
          save_path: 保存文件的完整路径（字符串）
        """
        if len(self.current_image.shape) != 2:
            raise ValueError("只能保存灰度图为RAW格式")
        
        with open(save_path, 'wb') as file:
            height, width = self.current_image.shape
            file.write(struct.pack('II', width, height))
            file.write(self.current_image.tobytes())
    
    def save_png_image(self, save_path):
        """
        保存彩色图为PNG格式。
        
        参数:
          save_path: 保存文件的完整路径（字符串）
        """
        pil_image = create_pil_image(self.current_image)
        pil_image.save(save_path)


# ========== 程序入口 ==========
if __name__ == "__main__":
    # 创建tkinter根窗口
    root = tk.Tk()
    # 实例化图像浏览应用
    app = ImageViewerApp(root)
    # 启动GUI主循环，使窗口保持显示状态
    root.mainloop()
