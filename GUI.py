import logging
import os
import sys
import traceback

from PyQt5.Qt import QLabel, QPushButton, QPalette, QBrush, QStatusBar, QWidget, QApplication, QSlider, QFileDialog, QPainter, QDesktopWidget
from PyQt5.QtGui import QPixmap, QIcon, QColor, QCursor, QImage, QFont, QPen
from PyQt5.QtCore import QPoint, Qt

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import imageio.v2 as imageio

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

LOGGER = logging.getLogger(__name__)


class InPaintingMainWindow(QWidget):
    def __init__(self,
                 cfg_path='./configs/prediction/default.yaml'):  # E:/实践/图像修复/LAMA_Ceiling/lama/configs/prediction/default.yaml
        super(InPaintingMainWindow, self).__init__()
        ''' 设置主窗口相关参数 self '''
        # 绝对布局：空间尺寸设置
        window_width = 1200
        window_height = 800
        self.setFixedSize(window_width, window_height)
        self.centralization()
        # 设置背景图片
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./Materials/Pictures/Cloud1.jpg")))
        self.setPalette(palette)
        # 设置主窗口标题
        self.setWindowTitle("图像修复测试窗口")
        # 设置窗口图标
        self.setWindowIcon(QIcon(QPixmap("./Materials/Icons/Inpainting.svg")))

        ''' 设置状态栏相关参数 statusBar '''
        # 初始化状态栏
        self.statusBar = QStatusBar(self)
        self.statusBar.setFixedSize(200, 20)
        self.statusBar.move(00, window_height - 20)
        self.statusBar.setStyleSheet('QStatusBar{color:rgb(20, 20, 20,);}')
        self.statusBar.showMessage("欢迎使用CeiLing‘s InPainting", 3000)

        ''' 设置显示图片窗口 in; out'''
        show_width = window_width // 2 - 30
        show_height = window_width // 2 - 30
        self.show_width = show_width
        self.show_height = show_height

        # 设置输入图像窗口 (需要绘图) in_label
        self.in_label = QLabel(self)
        self.in_label.setFixedSize(show_width, show_height)
        self.in_label.move(20, 60)
        self.in_label.setAlignment(Qt.AlignCenter)
        canvas = QPixmap(show_width, show_height)
        canvas.fill(QColor(112, 112, 112, 40))
        self.in_label.setPixmap(canvas)
        cursor = QCursor(QPixmap("./Materials/Icons/Circle.svg"), 0, 0)
        self.in_label.setCursor(cursor)
        # 设置输入图像标题 in_title
        self.in_title = QLabel(self)
        self.in_title.setFixedSize(show_width, 30)
        self.in_title.move(20, 30)
        self.in_title.setStyleSheet("QLabel{font-size:16px; font-weight:bold; font-family:Arial;"
                                    "color:rgb(192, 128, 128); background-color:rgba(56, 56, 56, 128);}")
        self.in_title.setText("待修复图像")
        self.in_title.setAlignment(Qt.AlignCenter)

        # 设置输出图像窗口 out_label
        self.out_label = QLabel(self)
        self.out_label.setFixedSize(show_width, show_height)
        self.out_label.move(window_width - 20 - show_width, 60)
        self.out_label.setAlignment(Qt.AlignCenter)
        canvas = QPixmap(show_width, show_height)
        canvas.fill(QColor(112, 112, 112, 40))
        self.out_label.setPixmap(canvas)
        # 设置输入图像标题 out_title
        self.out_title = QLabel(self)
        self.out_title.setFixedSize(show_width, 30)
        self.out_title.move(window_width - 20 - show_width, 30)
        self.out_title.setStyleSheet("QLabel{font-size:16px; font-weight:bold; font-family:Arial;"
                                     "color:rgb(192, 128, 128); background-color:rgba(56, 56, 56, 128);}")
        self.out_title.setText("图像修复结果")
        self.out_title.setAlignment(Qt.AlignCenter)

        ''' 按钮设置 load/save/inpaint/exit '''
        button_height = 40
        button_width = 150
        # 加载图片按钮 load_file_btn
        self.load_file_btn = QPushButton(self)
        self.load_file_btn.setFixedSize(button_width, button_height)
        self.load_file_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(56, 56, 112,); background-color:rgba(56, 56, 56, 10)}"
                                         "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(112, 112, 224,); background-color:rgba(56, 56, 56, 10)}"
                                         )
        self.load_file_btn.setIcon(QIcon(QPixmap("./Materials/Icons/File.svg")))
        self.load_file_btn.setText("加载图片")
        self.load_file_btn.move(130, 80 + show_height)
        self.load_file_btn.clicked.connect(self.loadFile)

        # 图像保存按钮 save_file_btn
        self.save_file_btn = QPushButton(self)
        self.save_file_btn.setFixedSize(button_width, button_height)
        self.save_file_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(56, 56, 112,); background-color:rgba(56, 56, 56, 10)}"
                                         "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                         "color:rgb(112, 112, 224,); background-color:rgba(56, 56, 56, 10)}"
                                         )
        self.save_file_btn.setIcon(QIcon(QPixmap("./Materials/Icons/Save.svg")))
        self.save_file_btn.setText("保存图片")
        self.save_file_btn.move(380, 80 + show_height)
        self.save_file_btn.clicked.connect(self.saveImage)

        # 图像修复按钮 inpaint_btn
        self.inpaint_btn = QPushButton(self)
        self.inpaint_btn.setFixedSize(button_width, button_height)
        self.inpaint_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                       "color:rgb(56, 56, 112,); background-color:rgba(56, 56, 56, 10)}"
                                       "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                       "color:rgb(112, 112, 224,); background-color:rgba(56, 56, 56, 10)}"
                                       )
        self.inpaint_btn.setIcon(QIcon(QPixmap("./Materials/Icons/Pen.svg")))
        self.inpaint_btn.setText("图像修复")
        self.inpaint_btn.move(680, 80 + show_height)
        self.inpaint_btn.clicked.connect(self.Lama)

        # 退出程序 exit_btn
        self.exit_btn = QPushButton(self)
        self.exit_btn.setFixedSize(button_width, button_height)
        self.exit_btn.setStyleSheet("QPushButton{font-size:18px; font-family:Arial; font-weight:bold;"
                                    "color:rgb(56, 56, 112,); background-color:rgba(56, 56, 56, 10)}"
                                    "QPushButton:hover{font-size:20px; font-family:Arial; font-weight:bold;"
                                    "color:rgb(112, 112, 224,); background-color:rgba(56, 56, 56, 10)}"
                                    )
        self.exit_btn.setIcon(QIcon(QPixmap("./Materials/Icons/Close.svg")))
        self.exit_btn.setText("退出程序")
        self.exit_btn.move(920, 80 + show_height)
        self.exit_btn.clicked.connect(QApplication.instance().quit)

        ''' 滑块条显示及其参数设置 '''
        self.slider_label = QLabel(self)
        self.slider_label.setFixedSize(200, 30)
        self.slider_label.move(500, 700)
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider_label.setFont(QFont('Arial', 10))
        canvas = QPixmap(show_width, show_height)
        canvas.fill(QColor(112, 112, 112, 0))
        self.slider_label.setPixmap(canvas)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setParent(self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(15)
        self.slider.setSingleStep(1)
        self.slider.setValue(6)
        self.slider.setTickPosition(QSlider.TicksLeft)
        self.slider.setTickInterval(2)
        self.slider.setFixedSize(200, 20)
        self.slider.move(500, 730)
        self.slider_label.setText("笔触尺寸："+str(self.slider.value()))
        self.slider.valueChanged.connect(self.valueChange)

        ''' 模型及相关参数设置 '''
        with open(cfg_path, 'r',  encoding='utf-8') as f:
            self.predict_config = OmegaConf.create(yaml.safe_load(f))
        self.device = torch.device(self.predict_config.device)
        train_config_path = os.path.join(self.predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            self.train_config = OmegaConf.create(yaml.safe_load(f))

        self.train_config.training_model.predict_only = True
        self.train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(self.predict_config.model.path,
                                       'models',
                                       self.predict_config.model.checkpoint)
        print(checkpoint_path)
        self.model = load_checkpoint(self.train_config, checkpoint_path, strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)

        if not self.predict_config.indir.endswith('/'):
            self.predict_config.indir += '/'

        ''' 初始化参数 '''
        self.in_name = None
        self.in_image = None
        self.out_image = None
        self.lastPoint = QPoint()
        self.endPoint = QPoint()

    def centralization(self):
        # 设置窗口中心化
        screen = QDesktopWidget().screenGeometry()
        window = self.geometry()
        w = (screen.width() - window.width()) // 2
        h = (screen.height() - window.height()) // 2 - 50
        self.move(w, h)

    def loadFile(self):
        image_path, file_type = QFileDialog.getOpenFileName(self, "图像文件读取", "./", "*.PNG;;*.JPG;;*.png;;*.jpg")
        if image_path == '':
            self.statusBar.showMessage("加载图片失败", 2000)
        else:
            self.in_name = image_path.split('/')[-1]
            self.in_image = imageio.imread(image_path)
            size = self.in_image.shape
            h, w = size[0], size[1]
            if max([h, w]) > self.show_width - 20:
                ratio_h = 1.0 * (self.show_width - 20) / h
                ratio_w = 1.0 * (self.show_width - 20) / w
                ratio = min([ratio_h, ratio_w])
                h = int(ratio * h)
                w = int(ratio * w)
                self.in_image = cv2.resize(self.in_image, (w, h))
            self.in_image[self.in_image == 255] = 250
            bg = np.array([56, 56, 56, 32]) * np.ones((self.show_height, self.show_width, 4), dtype=np.uint8)
            top = self.show_height // 2 - h // 2
            down = top + h
            left = self.show_width // 2 - w // 2
            right = left + w
            bg[top:down, left:right, 0:3] = self.in_image
            bg[top:down, left:right, 3] = 255
            if os.path.exists("./temp") is False:
                os.makedirs("./temp", exist_ok=True)
            imageio.imwrite("./temp/in.png", bg)
            self.in_label.setPixmap(QPixmap("./temp/in.png"))
            canvas = QPixmap(self.show_width, self.show_height)
            canvas.fill(QColor(112, 112, 112, 40))
            self.out_label.setPixmap(canvas)
            self.statusBar.showMessage("成功加载图像", 2000)

    def Lama(self):
        if self.in_image is None:
            self.statusBar.showMessage("图像修复失败", 2000)
            pass
        else:
            image = self.in_label.pixmap().toImage()
            size = image.size()
            s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB
            arr = np.frombuffer(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
            arr = arr[:, :, ::-1]
            image = arr[:, :, 1:4]
            size = self.in_image.shape
            h, w = size[0], size[1]
            top = self.show_height // 2 - h // 2
            down = top + h
            left = self.show_width // 2 - w // 2
            right = left + w
            mask = image[top:down, left:right]
            if len(mask.shape) == 3:
                mask = np.max(mask, axis=-1)

            if os.path.exists("./lama") is False:
                os.makedirs("./lama", exist_ok=True)
            imageio.imwrite("./lama/image.png", self.in_image)
            mask = np.uint8(mask == 255)
            imageio.imwrite("./lama/image_mask.png", mask)

            out_ext = self.predict_config.get('out_ext', '.png')
            dataset = make_default_val_dataset(self.predict_config.indir, **self.predict_config.dataset)
            with torch.no_grad():
                mask_fname = dataset.mask_filenames[0]
                cur_out_fname = os.path.join(
                    self.predict_config.outdir,
                    os.path.splitext(mask_fname[len(self.predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
                batch = move_to_device(default_collate([dataset[0]]), self.device)

                batch['mask'] = (batch['mask'] > 0) * 1
                batch = self.model(batch)
                cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                imageio.imwrite(cur_out_fname, cur_res)
            self.out_image = cur_res
            bg = np.array([56, 56, 56, 32]) * np.ones((self.show_height, self.show_width, 4), dtype=np.uint8)
            bg[top:down, left:right, 0:3] = self.out_image
            bg[top:down, left:right, 3] = 255
            imageio.imwrite("./temp/out.png", bg)
            self.out_label.setPixmap(QPixmap(QImage("./temp/out.png")))
            self.statusBar.showMessage("图像修复完成", 2000)

    def saveImage(self):
        if self.in_name is not None:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Image", self.in_name, "All Files (*)",)
        else:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Image", './image.png', "All Files (*)",)
        if self.out_image is None:
            self.statusBar.showMessage("保存失败", 3000)
            pass
        elif filename == "":
            self.statusBar.showMessage("保存失败", 3000)
            pass
        else:
            imageio.imwrite(filename, self.out_image)
            self.statusBar.showMessage("保存成功", 3000)

    def valueChange(self):
        sender = self.sender()
        self.slider_label.setText("笔触尺寸：" + str(self.slider.value()))
        self.statusBar.showMessage("修改笔触尺寸为：{}".format(self.slider.value()), 2000)

    def paintEvent(self, event):
        pp = QPainter(self.in_label.pixmap())
        pp.setPen(QPen(Qt.white, self.slider.value(), Qt.DashLine))
        # 根据鼠标指针前后两个位置绘制直线
        pp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(20, 60, self.in_label.pixmap())

    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = QPoint(event.pos().x()-8, event.pos().y()-48)
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = QPoint(event.pos().x()-8, event.pos().y()-48)
            # 进行重新绘制
            self.update()
            self.statusBar.showMessage("", 3000)

    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = QPoint(event.pos().x()-8, event.pos().y()-48)
            self.update()


def main():
    app = QApplication(sys.argv)
    window = InPaintingMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
