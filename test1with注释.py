import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import colorsys

# --------------------------
# 核心图像生成函数
# --------------------------
def create_image_from_array(matrix, is_color=False):
    """将矩阵转换为PIL图像对象
    参数：
        matrix - 灰度图（二维numpy数组）或彩色图（三维numpy数组）
        is_color - 是否为彩色图像（自动检测时可设置为None）
    """
    # 自动检测模式，如果矩阵为三阶，则True
    if is_color is None:
        is_color = (len(matrix.shape) == 3)
    
    # 把矩阵转换为PIL图像
    if is_color:
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
    """③ 横坐标映射灰度图。取所有行grad[:,x]，x*255/511是为了把x从【0，511】映射到【0，255】区间"""
    grad = np.zeros((512, 512), dtype=np.uint8)
    for x in range(512):
        grad[:, x] = int(x * 255 / 511)
    return grad

def generate_rainbow():
    """④ 全彩色渐变图"""
    """
    生成一张 512×512 的彩色渐变图，从左到右颜色变化，形成 彩虹渐变效果
    RGB矩阵直接做色彩转换还是太难用了。这里我让大模型取了个巧，用了colorsys库，方便渐变与色彩转换。
    colorsys的格式是HSV，色相Hue-饱和度Saturation-亮度Value，后两项设置为1————纯色和最亮的颜色，主要用到了色相Hue。
    色相值从0-1依次为：红→黄→绿→青→蓝→紫→红
    先创建一个512*512*3的空数组（彩色图像，RGB三通道，dtype=np.uint8 表示像素值范围在 0~255），
    接着遍历512列，每列分配一个归一化的色相（x/512），再将HSV格式色相转换回归一化的RGB。
    最后把归一化（0.0~1.0）的RGB转为 0~255 的整数，填充一整列

    """
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for x in range(512):
        hue = x / 511
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        arr[:, x] = [int(c*255) for c in rgb]
    return arr

def generate_stripes(height=512):
    """⑤ 竖直灰度条纹图"""
    grays = [0, 31, 63, 95, 127, 159, 191, 224, 255]
    width = 32 * len(grays)  # 每个条纹宽32像素
    matrix = np.zeros((height, width), dtype=np.uint8)
    '''解释一下循环：enumerate(grays) 返回 (索引, 元素值),即i和gray灰度值'''
    for i, gray in enumerate(grays):
        x_start = i * 32
        x_end = (i+1) * 32
        matrix[:, x_start:x_end] = gray
    
    return matrix

# --------------------------
# GUI界面类
# --------------------------
'''定义一个名为 ImageDisplayApp 的类，方便操作，用来构建和控制整个 GUI 界面'''
class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("实验一：图像显示")
        
        # 当前显示的图像引用，用于存储当前显示的图像，后面会用到
        self.current_image = None
        
        # 调用 create_widgets() 方法，用于创建界面上的按钮和标签等控件，看下面
        self.create_widgets()
    
    def create_widgets(self):
        """创建界面布局"""
        # 顶部按钮区域
        button_frame = ttk.Frame(self.root)#创建一个框架（组件），用来放按钮，后面会定义这个按钮
        button_frame.pack(pady=10)#设置宽度，即y轴方向的间距为 10。
        
        # 创建五个按钮
        buttons = [
            ("全黑图", self.show_black),
            ("全白图", self.show_white),
            ("横坐标渐变", self.show_gradient),
            ("彩虹渐变", self.show_rainbow),
            ("灰度条纹", self.show_stripes)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(button_frame, text=text, command=command)
            #框架在上面button_frame里，在这里塞按钮；text是上面元组里的text，即按钮名称；command就是那五个点下去运行的函数，后面会定义
            btn.pack(side='left', padx=5)#按钮从左到右，间距5像素
        
        # 图像显示区域
        self.display_frame = ttk.Frame(self.root)#创建一个显示图像的框架（组件），用来放图像
        self.display_frame.pack(expand=True, fill='both', padx=20, pady=20)#expand自动填充，fill水平垂直，pad是边距，设置为20像素
        
        # 初始占位标签#
        self.label = ttk.Label(self.display_frame, text="点击按钮显示图像")
        self.label.pack(expand=True)#创建一个标签填充窗口，保证窗口中总有可见内容。没啥用。
    
    def show_image(self, matrix, is_color):
        """显示图像的核心方法"""
        # 清除旧内容（图像）
        for widget in self.display_frame.winfo_children():#获取 display_frame 内的所有组件
            widget.destroy()#删除所有组件
        
        # 生成PIL图像
        pil_image = create_image_from_array(matrix, is_color)
        
        # 把PIL图像转换为Tkinter图像
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # 创建Canvas显示图像
        canvas = tk.Canvas(self.display_frame, 
                          width=pil_image.width, 
                          height=pil_image.height)#画布大小与图像匹配
        canvas.create_image(0, 0, anchor='nw', image=tk_image)#锚点为左上角
        canvas.pack(expand=True)#让canvas填充整个框架
        
        # 保持图像引用
        '''
        Tkinter 需要手动保存 PhotoImage 的引用，否则 Python 垃圾回收 会清理 tk_image，导致界面上的图像 瞬间消失。
        写个 self.current_image 变量 防止图像被回收，确保 GUI 显示稳定。
        '''
        self.current_image = tk_image
    
    # --------------------------
    # 按钮回调方法。这里就是调用前面写好的函数来。
    # --------------------------
    def show_black(self):
        matrix = generate_black()
        self.show_image(matrix, is_color=False)
    
    def show_white(self):
        matrix = generate_white()
        self.show_image(matrix, is_color=False)
    
    def show_gradient(self):
        matrix = generate_gradient()
        self.show_image(matrix, is_color=False)
    
    def show_rainbow(self):
        matrix = generate_rainbow()
        self.show_image(matrix, is_color=True)
    
    def show_stripes(self):
        matrix = generate_stripes(height=256)  # 可以自定义高度，暂定256像素。我没搞明白题目要求
        self.show_image(matrix, is_color=False)

# --------------------------
# 启动程序
# --------------------------
if __name__ == "__main__":#判断当前脚本是否作为主程序执行（而非被导入其他模块）。这个写过python的都知道
    root = tk.Tk()
    app = ImageDisplayApp(root)#创建 ImageDisplayApp 类的实例，传入根窗口，初始化整个 GUI 界面。
    root.geometry("800x600")
    root.mainloop()#窗口主循环