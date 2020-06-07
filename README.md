# 基于Paddle的截图&OCR文字识别的实现

一款**截图识别文字**的OCR工具主要涉及2个环境：

- 截图
- OCR识别

## 前要

### OCR的应用场景

根据OCR的应用场景而言，我们可以大致分成识别特定场景下的专用OCR以及识别多种场景下的通用OCR。就前者而言，证件识别以及车牌识别就是专用OCR的典型案例。针对特定场景进行设计、优化以达到最好的特定场景下的效果展示。那通用的OCR就是使用在更多、更复杂的场景下，拥有比较好的泛性。在这个过程中由于场景的不确定性，比如：图片背景极其丰富、亮度不均衡、光照不均衡、残缺遮挡、文字扭曲、字体多样等等问题，会带来极大的挑战。

### OCR的技术路线

典型的OCR技术路线如下图所示：

![img](https://ai-studio-static-online.cdn.bcebos.com/d9d8d4f77c2a408fbac5288c3f7e729425142e076561498d8191c94394be8a8c)

其中OCR识别的关键路径在于文字检测和文本识别部分，这也是深度学习技术可以充分发挥功效的地方。PaddleHub为大家开源的预训练模型的网络结构是Differentiable Binarization+ CRNN，基于icdar2015数据集下进行的训练。

首先，DB是一种基于分割的文本检测算法。在各种文本检测算法中，基于分割的检测算法可以更好地处理弯曲等不规则形状文本，因此往往能取得更好的检测效果。但分割法后处理步骤中将分割结果转化为检测框的流程复杂，耗时严重。因此作者提出一个可微的二值化模块（Differentiable Binarization，简称DB），将二值化阈值加入训练中学习，可以获得更准确的检测边界，从而简化后处理流程。DB算法最终在5个数据集上达到了state-of-art的效果和性能。参考论文：[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)

下图是DB算法的结构图：

![img](https://ai-studio-static-online.cdn.bcebos.com/9de7da04065e4f3aba87e5d8b0191902e6b36c5aaf17452b8aa07af65e788659)

接着，我们使用 CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络，是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。参考论文:[An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)

下图是CRNN的网络结构图：

![img](https://ai-studio-static-online.cdn.bcebos.com/746765e409134b2fae3aa253bb500c8b0b99722c05dc40449787410e71647f62)

## 截图工具

很多人会把它想的非常复杂，其实，Python中有很多可以实现截图的库或者函数，最常见的有三种方法。

### 一、Python调用windows API实现屏幕截图

### 二、使用PIL的ImageGrab模块

### 三、使用Selenium截图

而我们需要做到的事鼠标框选范围截图，因此我们采用PyQt5和PIL实现截图功能。

我们只需要把`鼠标框选`的`起点`和`终点`坐标传给grab方法就可以实现截图功能。

那么，现在问题就转化为**如何获取鼠标框选的起点和终点？**

Textshot通过调用PyQt5并继承QWidget来实现鼠标框选过程中的一些方法来获取框选的起点和终点。

Textshot继承和重写QWidget方法主要包括如下几个，

- `keyPressEvent(self, event)`：键盘响应函数
- `paintEvent(self, event)`：UI绘制函数
- `mousePressEvent(self, event)`：鼠标点击事件
- `mouseMoveEvent(self, event)`：鼠标移动事件
- `mouseReleaseEvent(self, event)`：鼠标释放事件

可以看出，上面重写的方法以及囊括了截图过程中涉及的各个动作，

- 点击鼠标
- 拖动、绘制截图框
- 释放鼠标

当然了，这一部分有现成的

[轮子]: https://github.com/ianzhao05/textshot?u=5722964389&amp;m=4508439520834016&amp;cu=3655689037

可以直接使用：

```python
class Snipper(QtWidgets.QWidget):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent=parent, flags=flags)

        self.setWindowTitle("TextShot")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Dialog
        )

        self.setWindowState(self.windowState() | Qt.WindowFullScreen)
        self.screen = QtGui.QScreen.grabWindow(
            QtWidgets.QApplication.primaryScreen(),
            QtWidgets.QApplication.desktop().winId(),
        )
        palette = QtGui.QPalette()
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(self.screen))
        self.setPalette(palette)

        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.start, self.end = QtCore.QPoint(), QtCore.QPoint()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            QtWidgets.QApplication.quit()

        return super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 100))
        painter.drawRect(0, 0, self.width(), self.height())

        if self.start == self.end:
            return super().paintEvent(event)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3))
        painter.setBrush(painter.background())
        painter.drawRect(QtCore.QRect(self.start, self.end))
        return super().paintEvent(event)

    def mousePressEvent(self, event):
        self.start = self.end = QtGui.QCursor.pos()
        self.update()
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.end = QtGui.QCursor.pos()
        self.update()
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start == self.end:
            return super().mouseReleaseEvent(event)

        self.hide()
        QtWidgets.QApplication.processEvents()
        shot = self.screen.copy(QtCore.QRect(self.start, self.end))
        processImage(shot)
        QtWidgets.QApplication.quit()


def processImage(img):

    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.ReadWrite)
    img.save(buffer, "PNG")
    pil_img = Image.open(io.BytesIO(buffer.data()))
    buffer.close()



if __name__ == '__main__':

    QtCore.QCoreApplication.setAttribute(Qt.AA_DisableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    snipper = Snipper(window)
    snipper.show()
    sys.exit(app.exec_())
```

## OCR文字识别

那么我们的文字识别模型选择了Paddle最新推出的OCR识别模型。改模型同时支持中英文识别；支持倾斜、竖排等多种方向文字识别。

识别文字算法采用CRNN (Convolutional Recurrent Neural Network)即卷积递归神经网络。其是DCNN和RNN的组合，专门用于识别图像中的序列式对象。与CTC loss配合使用，进行文字识别，可以直接从文本词级或行级的标注中学习，不需要详细的字符级的标注。该Module是一个通用的OCR模型，支持直接预测。

这一步我们就要做的是将截取的图片传入文字识别模型即可。

```python
import os
os.environ['HUB_HOME'] = "./modules"
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PIL import Image
import io
import sys
import numpy as np
import paddlehub as hub


class Snipper(QtWidgets.QWidget):
    def __init__(self, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent=parent, flags=flags)

        self.setWindowTitle("TextShot")
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Dialog
        )

        self.setWindowState(self.windowState() | Qt.WindowFullScreen)
        self.screen = QtGui.QScreen.grabWindow(
            QtWidgets.QApplication.primaryScreen(),
            QtWidgets.QApplication.desktop().winId(),
        )
        palette = QtGui.QPalette()
        palette.setBrush(self.backgroundRole(), QtGui.QBrush(self.screen))
        self.setPalette(palette)

        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.start, self.end = QtCore.QPoint(), QtCore.QPoint()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            QtWidgets.QApplication.quit()

        return super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 100))
        painter.drawRect(0, 0, self.width(), self.height())

        if self.start == self.end:
            return super().paintEvent(event)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3))
        painter.setBrush(painter.background())
        painter.drawRect(QtCore.QRect(self.start, self.end))
        return super().paintEvent(event)

    def mousePressEvent(self, event):
        self.start = self.end = QtGui.QCursor.pos()
        self.update()
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.end = QtGui.QCursor.pos()
        self.update()
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start == self.end:
            return super().mouseReleaseEvent(event)

        self.hide()
        QtWidgets.QApplication.processEvents()
        shot = self.screen.copy(QtCore.QRect(self.start, self.end))
        processImage(shot)
        QtWidgets.QApplication.quit()


def processImage(img):

    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.ReadWrite)
    img.save(buffer, "PNG")
    pil_img = Image.open(io.BytesIO(buffer.data()))
    buffer.close()

    np_images = [np.array(pil_img)]

    results = ocr.recognize_text(
        images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
        use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
        visualization=True,  # 是否将识别结果保存为图片文件；
        box_thresh=0.5,  # 检测文本框置信度的阈值；
        text_thresh=0.5)  # 识别中文文本置信度的阈值；

    text = []

    for result in results:
        data = result['data']
        save_path = result['save_path']
        for infomation in data:
            print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ',
                  infomation['text_box_position'])
            text.append(str(infomation['text']) + '\n')

    print(text)

    with open('data.txt', 'w') as f:
        for i in text:
            f.write(str(i))

    os.system(r'data.txt')


if __name__ == '__main__':
    # 加载移动端预训练模型
    # ocr = hub.Module(name="chinese_ocr_db_crnn_mobile")
    # 服务端可以加载大模型，效果更好
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")

    QtCore.QCoreApplication.setAttribute(Qt.AA_DisableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    snipper = Snipper(window)
    snipper.show()
    sys.exit(app.exec_())

```

那么我们可以测试一下它的效果：



![QQ截图20200605224852](https://ai-studio-static-online.cdn.bcebos.com/de1e5cfd705c4ab6b35245f1d0f60a227b07295a75a54be48e3da4c41079d021)

![QQ截图20200605224932](https://ai-studio-static-online.cdn.bcebos.com/2503319942a6470c8d5b181e705a5da9951af86b75d943dcbc7b75c1566a8ff9)

![ndarray_1591368477.4426317](https://ai-studio-static-online.cdn.bcebos.com/e65d704b6bc74edeb036fa0d666bfba5d9ae9cc0a8ab414d995856163b6ebdcd)

那么再看一些模型的其他应用吧：

![img](https://ai-studio-static-online.cdn.bcebos.com/4cb2494ae11e4e02adb9bf6bcb504b3bd7cbe1bad52b4be581bd275c057b2583)

![img](https://ai-studio-static-online.cdn.bcebos.com/d515f5cfe09f46c6bdc5dd79a48013da9642d65aae0847149ae0a8b61b41beb7)

![img](https://ai-studio-static-online.cdn.bcebos.com/dfa1761756af4ccdaf5afedffb7a0c0745c0629164ba4cf8b1a61012c8fb6025)

更多详情可以查看：https://www.paddlepaddle.org.cn/hub/scene/ocr

截图&OCR项目地址：https://github.com/chenqianhe/-screenshot_and_ocr-