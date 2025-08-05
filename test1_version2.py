import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import colorsys

# --------------------------
# 核心图像生成函数
# --------------------------
def create_image_from_array(matrix):
    """将矩阵转换为PIL图像对象，自动判断彩色/灰度"""
    # 根据数组维度判断是否为彩色图像（三维数组且第三维度为3）
    if len(matrix.shape) == 3 and matrix.shape[2] == 3:
        return Image.fromarray(matrix.astype(np.uint8), 'RGB')
    else:
        return Image.fromarray(matrix.astype(np.uint8), 'L')

# --------------------------
# 图像矩阵生成函数
# --------------------------
def generate_black():
    """① 全黑（全0矩阵）灰度图"""
    return np.zeros((512, 512), dtype=np.uint8)

def generate_white():
    """② 全白（全255矩阵）灰度图"""
    return np.full((512, 512), 255, dtype=np.uint8)

def generate_gradient():
    """③ 横坐标映射灰度图"""
    grad = np.zeros((512, 512), dtype=np.uint8)
    for x in range(512):
        grad[:, x] = int(x * 255 / 511)
    return grad

def generate_rainbow():
    """④ 全彩色渐变图"""
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for x in range(512):
        hue = x / 511
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        arr[:, x] = [int(c*255) for c in rgb]
    return arr

def generate_stripes(height=512):
    """⑤ 竖直灰度条纹图"""
    grays = [0, 31, 63, 95, 127, 159, 191, 224, 255]
    width = 32 * len(grays)
    matrix = np.zeros((height, width), dtype=np.uint8)
    for i, gray in enumerate(grays):
        x_start = i * 32
        x_end = (i+1) * 32
        matrix[:, x_start:x_end] = gray
    return matrix

# --------------------------
# GUI界面类
# --------------------------
class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实验一：图像显示")
        self.current_image = None
        self.create_widgets()

    def create_widgets(self):
        """创建界面布局"""
        # 顶部按钮区域
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        buttons = [
            ("全黑图", self.show_black),
            ("全白图", self.show_white),
            ("横坐标渐变", self.show_gradient),
            ("彩虹渐变", self.show_rainbow),
            ("灰度条纹", self.show_stripes)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(button_frame, text=text, command=command)
            btn.pack(side='left', padx=5)
        
        # 图像显示区域
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # 初始占位标签
        self.label = ttk.Label(self.display_frame, text="点击按钮显示图像")
        self.label.pack(expand=True)

    def show_image(self, matrix):
        """显示图像的核心方法"""
        # 清除旧内容
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        # 生成并转换图像
        pil_image = create_image_from_array(matrix)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # 创建Canvas显示图像
        canvas = tk.Canvas(self.display_frame, 
                          width=pil_image.width, 
                          height=pil_image.height)
        canvas.create_image(0, 0, anchor='nw', image=tk_image)
        canvas.pack(expand=True)
        
        # 保持图像引用
        self.current_image = tk_image

    # --------------------------
    # 按钮回调方法
    # --------------------------
    def show_black(self):
        self.show_image(generate_black())
    
    def show_white(self):
        self.show_image(generate_white())
    
    def show_gradient(self):
        self.show_image(generate_gradient())
    
    def show_rainbow(self):
        self.show_image(generate_rainbow())
    
    def show_stripes(self):
        self.show_image(generate_stripes(height=256))

# --------------------------
# 启动程序
# --------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDisplayApp(root)
    root.geometry("800x600")
    root.mainloop()