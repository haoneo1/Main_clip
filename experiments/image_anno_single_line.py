import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QListWidget, QSplitter, QAction, QMenuBar, QMenu,
                            QMessageBox, QGroupBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QRect
from natsort import natsorted

class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化变量
        self.image_path = None
        self.pixmap = None
        self.original_size = (0, 0)  # 原始图片尺寸 (width, height)
        self.folder_path = ""
        self.image_files = []
        self.current_index = -1
        # 存储所有图片的标注，键为图片路径，值为直线坐标((x1,y1), (x2,y2)) - 原始像素坐标
        self.all_annotations = {}
        self.drawing = False
        self.start_point = QPoint()  # 显示坐标
        self.end_point = QPoint()    # 显示坐标
        self.scale_factor = 1.0      # 缩放因子：显示尺寸 / 原始尺寸
        
        # 设置窗口
        self.setWindowTitle("图像直线标注工具（原始像素坐标）")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建UI
        self.init_ui()
        
    def init_ui(self):
        # 创建菜单栏
        self.create_menu_bar()
        
        # 主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 顶部按钮布局 - 分为两组
        button_group = QHBoxLayout()
        
        # 第一组：文件夹和导航
        nav_group = QGroupBox("导航")
        nav_layout = QHBoxLayout()
        nav_group.setLayout(nav_layout)
        
        self.select_folder_btn = QPushButton("选择图片文件夹")
        self.select_folder_btn.clicked.connect(self.select_folder)
        nav_layout.addWidget(self.select_folder_btn)
        
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        self.prev_btn.setShortcut(Qt.Key.Key_Left)
        
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        self.next_btn.setShortcut(Qt.Key.Key_Right)
        
        # 第二组：标注操作
        annot_group = QGroupBox("标注操作")
        annot_layout = QHBoxLayout()
        annot_group.setLayout(annot_layout)
        
        self.clear_btn = QPushButton("清除当前标注")
        self.clear_btn.clicked.connect(self.clear_current_annotation)
        self.clear_btn.setEnabled(False)
        annot_layout.addWidget(self.clear_btn)
        
        self.save_current_btn = QPushButton("保存当前标注")
        self.save_current_btn.clicked.connect(self.save_current_annotation)
        self.save_current_btn.setEnabled(False)
        annot_layout.addWidget(self.save_current_btn)
        
        self.save_all_btn = QPushButton("保存所有标注")
        self.save_all_btn.clicked.connect(self.save_all_annotations)
        self.save_all_btn.setEnabled(False)
        annot_layout.addWidget(self.save_all_btn)
        
        button_group.addWidget(nav_group)
        button_group.addWidget(annot_group)
        main_layout.addLayout(button_group)
        
        # 分割器，用于图片显示和文件列表
        splitter = QSplitter(Qt.Horizontal)
        
        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 600)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        splitter.addWidget(self.image_label)
        
        # 文件列表
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_clicked)
        self.file_list.setMaximumWidth(250)
        splitter.addWidget(self.file_list)
        
        # 设置分割器比例
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter, stretch=1)
        
    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        
        # 打开文件夹动作
        open_folder_action = QAction('打开文件夹', self)
        open_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(open_folder_action)
        
        # 保存当前标注动作
        save_current_action = QAction('保存当前标注', self)
        save_current_action.triggered.connect(self.save_current_annotation)
        file_menu.addAction(save_current_action)
        
        # 保存所有标注动作
        save_all_action = QAction('保存所有标注', self)
        save_all_action.triggered.connect(self.save_all_annotations)
        file_menu.addAction(save_all_action)
        
        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu('编辑')
        
        # 清除当前标注动作
        clear_action = QAction('清除当前标注', self)
        clear_action.triggered.connect(self.clear_current_annotation)
        edit_menu.addAction(clear_action)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹", "")
        if folder:
            self.folder_path = folder
            self.image_files = []
            self.all_annotations = {}  # 重置标注字典
            
            # 加载图片文件
            self.load_image_files()
            
            # 尝试加载已有的标注文件
            self.load_all_annotations()
            
            self.current_index = 0
            if self.image_files:
                self.load_image(self.image_files[self.current_index])
                self.update_buttons()
                self.update_file_list()
    
    def load_image_files(self):
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        for file in os.listdir(self.folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                self.image_files.append(os.path.join(self.folder_path, file))
        
        #self.image_files.sort()
        self.image_files = natsorted(self.image_files)
    
    def update_file_list(self):
        self.file_list.clear()
        for file in self.image_files:
            item_text = os.path.basename(file)
            # 检查是否有标注
            if file in self.all_annotations and self.all_annotations[file] is not None:
                item_text += " (已标注)"
            self.file_list.addItem(item_text)
        
        if 0 <= self.current_index < len(self.image_files):
            self.file_list.setCurrentRow(self.current_index)
    
    def get_annotations_file_path(self):
        # 所有标注信息保存的文件路径
        if self.folder_path:
            return os.path.join(self.folder_path, "all_annotations.json")
        return "all_annotations.json"
    
    def load_all_annotations(self):
        """加载所有图片的标注信息"""
        annot_path = self.get_annotations_file_path()
        
        if os.path.exists(annot_path):
            try:
                with open(annot_path, 'r') as f:
                    data = json.load(f)
                
                # 验证并加载标注数据（这些是原始像素坐标）
                for img_path, line_data in data.items():
                    if os.path.exists(img_path) and img_path in self.image_files:
                        if line_data is None:
                            self.all_annotations[img_path] = None
                        elif isinstance(line_data, list) and len(line_data) == 2:
                            # 这些是原始像素坐标，直接保存
                            self.all_annotations[img_path] = (
                                (line_data[0][0], line_data[0][1]),
                                (line_data[1][0], line_data[1][1])
                            )
                
                QMessageBox.information(self, "成功", f"已加载 {len(self.all_annotations)} 条标注信息")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载标注文件失败: {str(e)}")
    
    def load_image(self, file_path):
        self.image_path = file_path
        self.pixmap = QPixmap(file_path)
        if self.pixmap.isNull():
            print(f"无法加载图片: {file_path}")
            return
        
        # 保存原始图片尺寸
        self.original_size = (self.pixmap.width(), self.pixmap.height())
        
        self.update_image_display()
        self.setWindowTitle(f"图像直线标注工具 - {os.path.basename(file_path)} "
                          f"({self.original_size[0]}x{self.original_size[1]})")
        self.update_buttons()
    
    def update_image_display(self):
        if self.pixmap:
            # 调整图像大小以适应窗口，但保持比例
            scaled_pixmap = self.pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.IgnoreAspectRatio, 
                Qt.SmoothTransformation
            )
            
            # 计算缩放因子：显示尺寸 / 原始尺寸
            self.scale_factor_x = scaled_pixmap.width() / self.original_size[0]
            self.scale_factor_y = scaled_pixmap.height() / self.original_size[1]
            
            # 创建一个可以绘制的图像副本
            self.temp_pixmap = scaled_pixmap.copy()
            
            # 绘制已有的直线
            painter = QPainter(self.temp_pixmap)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            
            # 从标注字典中获取当前图片的标注（原始坐标）
            current_line = self.all_annotations.get(self.image_path)
            if current_line:
                # 将原始坐标转换为显示坐标进行绘制
                p1 = QPoint(
                    int(current_line[0][0] * self.scale_factor_x),
                    int(current_line[0][1] * self.scale_factor_y)
                )
                p2 = QPoint(
                    int(current_line[1][0] * self.scale_factor_x),
                    int(current_line[1][1] * self.scale_factor_y)
                )
                painter.drawLine(p1, p2)
            
            painter.end()
            self.image_label.setPixmap(self.temp_pixmap)
    
    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image(self.image_files[self.current_index])
            self.file_list.setCurrentRow(self.current_index)
    
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_image(self.image_files[self.current_index])
            self.file_list.setCurrentRow(self.current_index)
    
    def update_buttons(self):
        has_images = len(self.image_files) > 0
        self.prev_btn.setEnabled(has_images and self.current_index > 0)
        self.next_btn.setEnabled(has_images and self.current_index < len(self.image_files) - 1)
        self.clear_btn.setEnabled(has_images)
        self.save_current_btn.setEnabled(has_images)
        self.save_all_btn.setEnabled(has_images)
    
    def on_file_clicked(self, item):
        index = self.file_list.row(item)
        if index != self.current_index and 0 <= index < len(self.image_files):
            self.current_index = index
            self.load_image(self.image_files[self.current_index])
    
    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton and self.pixmap:
            self.drawing = True
            self.start_point = event.pos() - self.image_label.pos()
    
    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.pixmap:
            self.drawing = False
            self.end_point = event.pos() - self.image_label.pos()
            
            # 确保起点和终点在图像范围内
            if self.is_point_in_image(self.start_point) and self.is_point_in_image(self.end_point):
                # 将显示坐标转换为原始像素坐标并保存
                original_start = (
                    int(self.start_point.x() / self.scale_factor_x),
                    int(self.start_point.y() / self.scale_factor_y)
                )
                original_end = (
                    int(self.end_point.x() / self.scale_factor_x),
                    int(self.end_point.y() / self.scale_factor_y)
                )
                
                self.all_annotations[self.image_path] = (original_start, original_end)
                self.update_image_display()
                self.update_file_list()  # 更新文件列表显示
    
    def mouse_move_event(self, event):
        if self.drawing and self.pixmap:
            self.end_point = event.pos()
            # 创建临时图像显示正在绘制的直线
            temp = self.temp_pixmap.copy()
            painter = QPainter(temp)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(self.start_point, self.end_point)
            painter.end()
            self.image_label.setPixmap(temp)
    
    def is_point_in_image(self, point):
        # 检查点是否在图像范围内
        if self.temp_pixmap:
            return QRect(0, 0, self.temp_pixmap.width(), self.temp_pixmap.height()).contains(point)
        return False
    
    def clear_current_annotation(self):
        """清除当前图片的标注"""
        if self.image_path in self.all_annotations:
            del self.all_annotations[self.image_path]
            self.update_image_display()
            self.update_file_list()
            QMessageBox.information(self, "提示", "已清除当前图片的标注")
    
    def save_current_annotation(self):
        """只保存当前图片的标注到总文件"""
        if not self.image_path:
            QMessageBox.information(self, "提示", "没有加载图片！")
            return
        
        # 检查当前图片是否有标注
        if self.image_path not in self.all_annotations or self.all_annotations[self.image_path] is None:
            QMessageBox.information(self, "提示", "当前图片没有标注可保存！")
            return
        
        try:
            # 保存所有标注（其实是更新总文件）
            self._save_annotations_to_file()
            QMessageBox.information(self, "成功", f"当前图片的标注已保存")
            self.update_file_list()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存标注失败:\n{str(e)}")
    
    def save_all_annotations(self):
        """保存所有图片的标注信息"""
        if not self.image_files:
            QMessageBox.information(self, "提示", "没有加载任何图片！")
            return
        
        # 统计有多少标注
        annotated_count = sum(1 for annot in self.all_annotations.values() if annot is not None)
        
        if annotated_count == 0:
            QMessageBox.information(self, "提示", "没有任何标注可保存！")
            return
        
        try:
            self._save_annotations_to_file()
            QMessageBox.information(self, "成功", 
                                  f"所有标注已保存，共 {annotated_count} 条标注\n"
                                  f"保存路径: {self.get_annotations_file_path()}")
            self.update_file_list()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存标注失败:\n{str(e)}")
    
    def _save_annotations_to_file(self):
        """将所有标注信息保存到文件的内部方法"""
        # 保存的是原始像素坐标
        save_data = {}
        for img_path, line in self.all_annotations.items():
            if line is not None:
                save_data[img_path] = [
                    [line[0][0], line[0][1]],  # 原始起点坐标
                    [line[1][0], line[1][1]]   # 原始终点坐标
                ]
            else:
                save_data[img_path] = None
        
        annot_path = self.get_annotations_file_path()
        
        with open(annot_path, 'w') as f:
            json.dump(save_data, f, indent=4)
    
    def resizeEvent(self, event):
        # 窗口大小改变时重新调整图像大小和缩放因子
        if self.pixmap:
            self.update_image_display()
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_())
