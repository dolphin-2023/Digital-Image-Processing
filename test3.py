"""
数字图像处理实验 - 直方图分析
作者：Harry Zhao
日期：2025-03-11
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os
import glob
import struct
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def read_raw_image(file_path):
    """读取自定义的RAW格式图片。"""
    with open(file_path, 'rb') as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"文件头字节数错误：期望8字节，实际读取到{len(header)}字节")
        width, height = struct.unpack('II', header)
        pixels = file.read()
        return np.frombuffer(pixels, dtype=np.uint8).reshape((height, width))

def create_pil_image(numpy_array):
    """将numpy数组转换为PIL库的图片对象。"""
    if len(numpy_array.shape) == 3:
        return Image.fromarray(numpy_array.astype(np.uint8), 'RGB')
    else:
        return Image.fromarray(numpy_array.astype(np.uint8), 'L')

def mybincount(arr, minlength=None):
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
    # 使用 np.add.at 进行向量化的索引累加
    np.add.at(counts, arr, 1)   
    return counts

def calculate_gray_histogram(image_data):
    """计算灰度图像的直方图。"""
    counts = mybincount(image_data.ravel(), minlength=256)
    return counts / image_data.size

def calculate_color_histogram(image_data):
    """计算彩色图像的直方图。"""
    red_channel = image_data[:, :, 0]
    green_channel = image_data[:, :, 1]
    blue_channel = image_data[:, :, 2]
    red_hist = mybincount(red_channel.ravel(), minlength=256) / red_channel.size
    green_hist = mybincount(green_channel.ravel(), minlength=256) / green_channel.size
    blue_hist = mybincount(blue_channel.ravel(), minlength=256) / blue_channel.size
    return red_hist, green_hist, blue_hist

class ImageViewerApp:
    """图像直方图分析器的图形界面类。"""
    def __init__(self, master):
        self.master = master
        self.image_paths = []
        self.current_index = -1
        self.current_image = None
        self.master.title("实验三直方图")
        self.master.geometry("1000x800")
        self.setup_ui()
    
    def setup_ui(self):
        button_frame = ttk.Frame(self.master)
        button_frame.pack(pady=10)
        buttons = [
            ("打开文件", self.open_file),
            ("打开目录", self.open_directory),
            ("上一张", self.show_previous),
            ("下一张", self.show_next),
            ("保存图片", self.save_image)
        ]
        for text, command in buttons:
            ttk.Button(button_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
        self.info_label = ttk.Label(self.master, text="图片信息：未加载")
        self.info_label.pack(pady=5)
        self.image_canvas = tk.Canvas(self.master, bg='#e0e0e0')
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_on_canvas = None
        self.histogram_frame = ttk.Frame(self.master)
        self.histogram_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.figure = Figure(figsize=(8, 3))
        self.hist_canvas = FigureCanvasTkAgg(self.figure, self.histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def open_file(self):
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
        directory = filedialog.askdirectory()
        if directory:
            self.load_images_from_directory(directory)
            if self.image_paths:
                self.current_index = 0
                self.show_current_image()
    
    def load_images_from_directory(self, directory):
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'raw']
        self.image_paths = []
        for ext in extensions:
            self.image_paths += glob.glob(os.path.join(directory, f"*.{ext}"))
        self.image_paths.sort()
        if not self.image_paths:
            messagebox.showinfo("提示", "该目录没有找到图片文件")
    
    def show_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return
        file_path = self.image_paths[self.current_index]
        try:
            if file_path.lower().endswith('.raw'):
                self.current_image = read_raw_image(file_path)
            else:
                pil_image = Image.open(file_path)
                self.current_image = np.array(pil_image)
            self.update_image_display()
            self.update_histogram()
            self.update_info_label()
        except Exception as e:
            messagebox.showerror("错误", f"无法读取文件：{str(e)}")
    
    def update_image_display(self):
        pil_image = create_pil_image(self.current_image)
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        resized_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(resized_image)
        self.image_canvas.delete("all")
        self.image_on_canvas = self.image_canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=tk_image)
        self.image_canvas.tk_image = tk_image
    
    def update_info_label(self):
        if self.current_image is None:
            return
        if len(self.current_image.shape) == 2:
            height, width = self.current_image.shape
            channels = 1
        else:
            height, width, channels = self.current_image.shape
        info_text = f"尺寸：{width}×{height} 像素 | 通道：{channels} | 当前：{self.current_index + 1}/{len(self.image_paths)}"
        self.info_label.config(text=info_text)
    
    def update_histogram(self):
        self.figure.clear()
        if len(self.current_image.shape) == 2:
            gray_hist = calculate_gray_histogram(self.current_image)
            ax = self.figure.add_subplot(111)
            ax.bar(range(256), gray_hist, width=1.0, color='gray')
            ax.set_title("灰度直方图")
        else:
            r_hist, g_hist, b_hist = calculate_color_histogram(self.current_image)
            ax = self.figure.add_subplot(131)
            ax.bar(range(256), r_hist, width=1.0, color='red')
            ax.set_title("红色通道")
            ax = self.figure.add_subplot(132)
            ax.bar(range(256), g_hist, width=1.0, color='green')
            ax.set_title("绿色通道")
            ax = self.figure.add_subplot(133)
            ax.bar(range(256), b_hist, width=1.0, color='blue')
            ax.set_title("蓝色通道")
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
    
    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        try:
            original_dir = os.path.dirname(self.image_paths[self.current_index])
            save_dir = os.path.join(original_dir, "保存结果")
            os.makedirs(save_dir, exist_ok=True)
            original_name = os.path.basename(self.image_paths[self.current_index])
            if len(self.current_image.shape) == 2:
                save_path = os.path.join(save_dir, original_name)
                self.save_raw_image(save_path)
            else:
                base_name = os.path.splitext(original_name)[0]
                save_path = os.path.join(save_dir, f"{base_name}.png")
                self.save_png_image(save_path)
            messagebox.showinfo("保存成功", f"图片已保存到：\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存失败", f"错误原因：{str(e)}")
    
    def save_raw_image(self, save_path):
        if len(self.current_image.shape) != 2:
            raise ValueError("只能保存灰度图为RAW格式")
        with open(save_path, 'wb') as file:
            height, width = self.current_image.shape
            file.write(struct.pack('II', width, height))
            file.write(self.current_image.tobytes())
    
    def save_png_image(self, save_path):
        pil_image = create_pil_image(self.current_image)
        pil_image.save(save_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()