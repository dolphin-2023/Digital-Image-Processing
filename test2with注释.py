"""
实验二完整实现
功能：在实验一基础上扩展RAW文件处理、保存管理、图像导航功能
作者：Harry Zhao, Deepseek R1
日期：20250308
"""

# ======================
# 模块导入部分
# ======================
import tkinter as tk  # 基础GUI库，提供窗口、控件等
from tkinter import ttk  # 现代风格UI组件
from tkinter import filedialog  # 文件对话框功能
from tkinter import messagebox  # 消息提示框
from PIL import Image, ImageTk  # 图像处理库，支持多种格式
import numpy as np  # 科学计算库，处理矩阵数据
import colorsys  # 颜色空间转换（HSV->RGB）
import os  # 操作系统接口，处理文件路径
import glob  # 文件通配符匹配
import struct  # 二进制数据打包/解包

# ======================
# 图像处理核心函数
# ======================
def create_image_from_array(matrix):
    """
    将numpy数组转换为PIL图像对象，自动识别彩色/灰度
    参数：
        matrix: numpy数组，形状为(height, width)或(height, width, 3)
    返回：
        PIL.Image对象
    """
    # 判断是否为三维数组且第三维度为3（RGB彩色图像）
    if len(matrix.shape) == 3 and matrix.shape[2] == 3:
        # 转换为8位无符号整型，创建RGB图像
        return Image.fromarray(matrix.astype(np.uint8), 'RGB')
    else:
        # 单通道灰度图像
        return Image.fromarray(matrix.astype(np.uint8), 'L')

# ======================
# RAW文件处理函数（实验2核心）
# ======================
def read_raw(file_name):
    """
    读取自定义格式的RAW文件
    文件结构：
    - 前8字节：宽度（4字节无符号整数）+ 高度（4字节无符号整数）
    - 后续数据：灰度像素值（每个像素1字节，0-255）
    参数：
        file_name: 文件路径
    返回：
        numpy数组（height, width）
    异常：
        当文件头不完整或数据尺寸不匹配时抛出ValueError
    """
    with open(file_name, 'rb') as f:  # 以二进制只读模式打开文件
        # 读取文件头（前8字节）
        header = f.read(8)
        
        # 检查文件头完整性
        if len(header) != 8:
            raise ValueError("文件头损坏：需要8字节但仅读取到{}字节".format(len(header)))#raise用于抛出异常
        
        # 解析宽度和高度（'II'表示两个unsigned int，各占4字节）
        width, height = struct.unpack('II', header)
        
        # 读取全部像素数据
        pixel_data = f.read()
        
        # 验证数据量是否符合预期
        expected_bytes = width * height
        if len(pixel_data) != expected_bytes:
            raise ValueError(
                "数据不完整：预期{}字节，实际{}字节".format(expected_bytes, len(pixel_data)))
        
        # 将字节数据转换为numpy数组，并调整形状为(height, width)
        return np.frombuffer(pixel_data, dtype=np.uint8).reshape((height, width))

def write_raw(file_name, array):
    """
    将二维numpy数组保存为RAW文件
    参数：
        file_name: 输出文件路径
        array: 二维numpy数组（dtype应为uint8）
    异常：
        当输入不是二维数组时抛出ValueError
    """
    # 输入验证：必须是二维数组
    if array.ndim != 2:
        raise ValueError("无效的数组维度：需要2维，实际是{}维".format(array.ndim))
    
    # 获取图像尺寸（numpy的shape返回(height, width)）
    height, width = array.shape
    
    with open(file_name, 'wb') as f:  # 二进制写入模式
        # 写入文件头（宽度和高度各4字节）
        f.write(struct.pack('II', width, height))
        
        # 将numpy数组转换为字节流写入文件
        f.write(array.tobytes())#tobytes()可以将数组转换为连续的字节序列

# ======================
# 实验1的图像生成函数
# ======================
def generate_black():
    """生成512x512全黑灰度图（所有像素值为0）"""
    return np.zeros((512, 512), dtype=np.uint8)

def generate_white():
    """生成512x512全白灰度图（所有像素值为255）"""
    return np.full((512, 512), 255, dtype=np.uint8)

def generate_gradient():
    """生成水平渐变灰度图（从左到右0-255线性变化）"""
    grad = np.zeros((512, 512), dtype=np.uint8)
    for x in range(512):
        # 每列设置为对应灰度值（511保证最大值为255）
        grad[:, x] = int(x * 255 / 511)
    return grad

def generate_rainbow():
    """生成水平彩虹渐变彩色图（HSV颜色空间转换）"""
    arr = np.zeros((512, 512, 3), dtype=np.uint8)  # 创建三维数组（RGB）
    for x in range(512):
        hue = x / 511  # 色相值（0-1）
        # 将HSV转换为RGB（H范围0-1，S=1, V=1）
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # 将浮点数转换为0-255整数，并填充整列
        arr[:, x] = [int(c * 255) for c in rgb]
    return arr

def generate_stripes(height=512):
    """生成垂直灰度条纹图"""
    grays = [0, 31, 63, 95, 127, 159, 191, 224, 255]  # 预设灰度值
    stripe_width = 32  # 每个条纹的宽度
    total_width = stripe_width * len(grays)  # 计算总宽度
    matrix = np.zeros((height, total_width), dtype=np.uint8)
    
    # 填充每个条纹
    for i, gray_value in enumerate(grays):
        start_col = i * stripe_width
        end_col = (i + 1) * stripe_width
        matrix[:, start_col:end_col] = gray_value
        
    return matrix

# ======================
# GUI主类（完整注释）
# ======================
class ImageDisplayApp:
    def __init__(self, root):
        """
        初始化图像显示应用
        参数：
            root: Tkinter根窗口对象
        """
        # 窗口基础设置
        self.root = root
        self.root.title("图像处理实验二 - 完整注释版")  # 窗口标题
        self.root.geometry("800x600")  # 初始窗口大小
        
        # 状态管理变量
        self.current_image = None  # 当前显示的Tk图像对象（防止被垃圾回收）
        self.raw_images = []  # 存储加载的RAW文件列表，元素为元组（文件路径，像素矩阵）
        self.current_raw_index = -1  # 当前显示的RAW文件索引（-1表示无）
        self.navigation_enabled = False  # 是否启用图像导航模式
        self.save_enabled = tk.BooleanVar(value=True)  # 自动保存开关（默认开启）
        self.save_path = None  # 当前保存路径
        
        # 创建界面组件
        self.create_widgets()
        
        # 绑定全局键盘事件
        self.root.bind("<Escape>", lambda e: self.root.quit())  # ESC键退出程序

    def create_widgets(self):
        """构建用户界面布局"""
        # 控制面板区域（顶部按钮栏）
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, fill='x', padx=5)  # 填充X方向，上下边距10px
        
        # 功能按钮列表（文本，回调函数）
        button_config = [
            ("打开文件", self.open_image_file),
            ("加载RAW目录", self.load_raw_directory),
            ("保存设置", self.configure_saving),
            ("全黑图", self.show_black),
            ("全白图", self.show_white),
            ("横坐标渐变", self.show_gradient),
            ("彩虹渐变", self.show_rainbow),
            ("灰度条纹", self.show_stripes)
        ]
        
        # 动态创建按钮
        for text, command in button_config:
            btn = ttk.Button(control_frame, text=text, command=command)
            btn.pack(side='left', padx=2)  # 水平排列，间距2px
            
        # 保存控制组件（右侧区域）
        save_control_frame = ttk.Frame(control_frame)
        save_control_frame.pack(side='right', padx=10)
        
        # 自动保存复选框
        ttk.Checkbutton(
            save_control_frame, 
            text="自动保存",
            variable=self.save_enabled  # 绑定到BooleanVar变量
        ).pack(side='left')
        
        # 打开保存目录按钮
        ttk.Button(
            save_control_frame,
            text="打开保存目录",
            command=self.open_save_dir
        ).pack(side='left', padx=5)
        
        # 图像显示区域
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # 信息显示标签（显示图像参数和系统状态）
        self.info_label = ttk.Label(
            self.display_frame, 
            text="就绪 | 保存路径：未设置"
        )
        self.info_label.pack(side='top', fill='x', pady=5)
        
        # 初始占位标签（无图像时显示）
        self.image_placeholder = ttk.Label(
            self.display_frame, 
            text="请点击上方按钮显示图像"
        )
        self.image_placeholder.pack(expand=True)

    def update_info(self, message):
        """
        更新状态信息栏
        参数：
            message: 要显示的主要信息
        """
        # 构建完整信息字符串
        full_info = f"{message} | 保存路径：{self.save_path or '未设置'} | 自动保存：{'开启' if self.save_enabled.get() else '关闭'}"
        self.info_label.config(text=full_info)

    def show_image(self, matrix):
        """
        核心图像显示方法
        参数：
            matrix: numpy数组（灰度或彩色）
        """
        # 清除旧显示内容（保留信息标签）
        for widget in self.display_frame.winfo_children():
            if widget not in [self.info_label]:
                widget.destroy()
        
        # 生成PIL图像对象
        pil_image = create_image_from_array(matrix)
        
        # 转换为Tkinter兼容的图像对象
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # 创建Canvas画布用于显示图像
        self.canvas = tk.Canvas(
            self.display_frame,
            width=pil_image.width,  # 设置画布尺寸与图像一致
            height=pil_image.height
        )
        # 将图像放置在画布左上角
        self.canvas.create_image(0, 0, anchor='nw', image=tk_image)
        self.canvas.pack(expand=True)  # 填充可用空间
        
        # 保持图像对象的引用（防止被Python垃圾回收）
        self.current_image = tk_image
        
        # 更新状态信息
        shape_info = "x".join(map(str, matrix.shape))
        channel_info = "RGB" if matrix.ndim == 3 else "灰度"
        self.update_info(f"当前图像：{shape_info} | 模式：{channel_info}")
        
        # 如果处于导航模式，绑定鼠标事件
        if self.navigation_enabled:
            # 左键（Button-1）绑定下一张
            self.canvas.bind("<Button-1>", self.next_image)
            # 右键（Button-3）绑定上一张
            self.canvas.bind("<Button-3>", self.previous_image)
        else:
            # 解绑事件防止冲突
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<Button-3>")

    # ======================
    # 文件管理功能
    # ======================
    def load_raw_directory(self):
        """加载指定目录下的所有RAW文件"""
        # 弹出目录选择对话框
        selected_dir = filedialog.askdirectory(title="选择RAW文件所在目录")
        if not selected_dir:  # 用户取消选择
            return
        
        # 设置默认保存路径（原目录下的save子目录）
        self.save_path = os.path.join(selected_dir, "save")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)  # 递归创建目录
            self.update_info(f"已创建保存目录：{self.save_path}")
        
        # 使用通配符查找所有.raw文件
        raw_files = glob.glob(os.path.join(selected_dir, "*.raw"))
        if not raw_files:
            messagebox.showwarning("无文件", "所选目录中没有找到RAW文件")
            return
        
        # 清空之前的加载记录
        self.raw_images.clear()
        
        # 逐个加载文件
        for file_path in raw_files:
            try:
                # 调用read_raw读取文件
                pixel_matrix = read_raw(file_path)
                # 存储文件路径和像素矩阵
                self.raw_images.append((file_path, pixel_matrix))
            except Exception as e:
                # 显示错误提示但继续加载其他文件
                messagebox.showerror(
                    "加载失败",
                    f"文件：{os.path.basename(file_path)}\n错误：{str(e)}"
                )
        
        # 如果有成功加载的文件
        if self.raw_images:
            self.current_raw_index = 0  # 从第一个文件开始
            self.navigation_enabled = True  # 启用导航模式
            self.show_raw_image()  # 显示当前图像
            
            # 如果自动保存开启，执行保存操作
            if self.save_enabled.get():
                self.auto_save_images()

    def auto_save_images(self):
        """自动保存所有加载的RAW文件副本"""
        if not self.save_path:
            self.update_info("错误：保存路径未设置")
            return
        
        try:
            saved_count = 0
            for idx, (_, matrix) in enumerate(self.raw_images):
                # 生成序号文件名（如：copy_001.raw）
                filename = f"copy_{idx:03d}.raw"
                full_path = os.path.join(self.save_path, filename)
                
                # 调用写入函数
                write_raw(full_path, matrix)
                saved_count += 1
            
            # 显示保存结果
            self.update_info(f"成功保存{saved_count}个文件到：{self.save_path}")
            messagebox.showinfo(
                "保存完成",
                f"已保存{saved_count}个文件到：\n{self.save_path}"
            )
        except Exception as e:
            messagebox.showerror("保存错误", f"自动保存失败：{str(e)}")
            self.update_info("自动保存失败")

    def configure_saving(self):
        """修改保存路径"""
        new_path = filedialog.askdirectory(title="选择新的保存位置")
        if new_path:
            self.save_path = new_path
            self.update_info(f"保存路径已更新为：{self.save_path}")

    def open_save_dir(self):
        """在文件资源管理器中打开保存目录"""
        if self.save_path and os.path.isdir(self.save_path):
            os.startfile(self.save_path)  # Windows系统打开目录
        else:
            messagebox.showinfo("目录不存在", "保存目录尚未设置或已被删除")

    # ======================
    # 图像导航功能
    # ======================
    def next_image(self, event=None):
        """显示下一张图像（左键触发）"""
        if self.raw_images:
            # 循环递增索引（取模运算实现循环）
            self.current_raw_index = (self.current_raw_index + 1) % len(self.raw_images)
            self.show_raw_image()

    def previous_image(self, event=None):
        """显示上一张图像（右键触发）"""
        if self.raw_images:
            # 循环递减索引
            self.current_raw_index = (self.current_raw_index - 1) % len(self.raw_images)
            self.show_raw_image()

    def show_raw_image(self):
        """显示当前索引对应的RAW图像"""
        if 0 <= self.current_raw_index < len(self.raw_images):
            file_path, pixel_matrix = self.raw_images[self.current_raw_index]
            self.show_image(pixel_matrix)
            # 在状态栏显示文件名
            self.update_info(f"正在查看：{os.path.basename(file_path)}")

    # ======================
    # 实验1功能（保持兼容）
    # ======================
    def open_image_file(self):
        """打开单个图像文件"""
        # 设置文件类型过滤器
        file_types = [
            ("图像文件", "*.jpg;*.jpeg;*.png;*.bmp;*.raw"),
            ("所有文件", "*.*")
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if not file_path:  # 用户取消选择
            return
        
        try:
            if file_path.lower().endswith('.raw'):
                # 处理RAW文件
                pixel_matrix = read_raw(file_path)
            else:
                # 使用PIL打开其他格式
                img = Image.open(file_path)
                pixel_matrix = np.array(img)  # 转换为numpy数组
            
            # 退出导航模式
            self.navigation_enabled = False
            self.show_image(pixel_matrix)
        except Exception as e:
            messagebox.showerror("打开失败", f"无法读取文件：{str(e)}")
    # --------------------------
    # 按钮回调方法。这里就是调用前面写好的函数。具体注释详见实验一
    # --------------------------
    def show_black(self):
        """显示全黑图"""
        self.navigation_enabled = False
        self.show_image(generate_black())

    def show_white(self):
        """显示全白图"""
        self.navigation_enabled = False
        self.show_image(generate_white())
    
    def show_gradient(self):
        """显示水平渐变图"""
        self.navigation_enabled = False
        self.show_image(generate_gradient())
    
    def show_rainbow(self):
        """显示彩虹渐变图"""
        self.navigation_enabled = False
        self.show_image(generate_rainbow())
    
    def show_stripes(self):
        """显示灰度条纹图"""
        self.navigation_enabled = False
        self.show_image(generate_stripes(height=256))

# --------------------------
# 程序入口
# --------------------------
if __name__ == "__main__":#判断当前脚本是否作为主程序执行（而非被导入其他模块）
    root = tk.Tk()  # 创建主窗口
    app = ImageDisplayApp(root) #创建 ImageDisplayApp 类的实例，传入根窗口，初始化整个 GUI 界面。
    #root.geometry("1280x720")好像不设置大小也能用，注释掉
    root.mainloop()# 进入事件循环