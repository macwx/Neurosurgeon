import torch
import torch.nn as nn
from collections import abc
from typing import List, Callable, Optional

class YOLOv8(nn.Module):
    def __init__(self, num_classes: int = 80, input_channels: int = 3, width_mult: float = 1.0) -> None:
        """
        YOLOv8模型的初始化
        num_classes: 输出的类别数，默认为80类
        input_channels: 输入图像的通道数，默认是3（RGB图像）
        """
        super(YOLOv8, self).__init__()

        # 基础部分（Backbone）：使用一个简单的卷积结构（类似于MobileNet）
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # YOLO头（YOLO Head）：预测目标的类别和边界框
        self.yolo_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 4, kernel_size=1, stride=1)  # 输出类别数 + 4（边界框信息）
        )

        # 初始化层
        self.len_backbone = len(self.backbone)
        self.len_head = len(self.yolo_head)
        self.len = self.len_backbone + self.len_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # 特征提取
        x = self.yolo_head(x)  # 目标检测头部
        return x

    def __len__(self):
        return len(self.backbone) + len(self.yolo_head)

    def __iter__(self):
        """ 用于遍历YOLOv8模型的每一层 """
        return SentenceIterator(self.backbone, self.yolo_head)

    def __getitem__(self, item):
        try:
            if item < self.len_backbone:
                layer = self.backbone[item]
            else:
                layer = self.yolo_head[item - self.len_backbone]
        except IndexError:
            raise StopIteration()
        return layer


class SentenceIterator(abc.Iterator):
    """
    YOLOv8迭代器
    下面是YOLOv8网络的迭代参数调整
    将下面的设置传入到YOLOv8的__iter__中可以完成对于YOLOv8网络的层级遍历
    """
    def __init__(self, backbone, yolo_head):
        self.backbone = backbone
        self.yolo_head = yolo_head
        self._index = 0
        self.len_backbone = len(backbone)
        self.len_head = len(yolo_head)

    def __next__(self):
        try:
            if self._index < self.len_backbone:
                layer = self.backbone[self._index]
            else:
                layer = self.yolo_head[self._index - self.len_backbone]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer


# 以下为YOLOv8模型的简单测试代码
if __name__ == "__main__":
    # 初始化模型
    model = YOLOv8(num_classes=80)

    # 创建一个假的输入
    input_tensor = torch.randn(1, 3, 224, 224)

    # 前向传播
    output = model(input_tensor)

    # 打印输出形状
    print("Output shape:", output.shape)

    # 通过迭代器遍历模型的每一层
    for layer in model:
        print(layer)

    # 获取模型的第一个层
    first_layer = model[0]
    print("First layer:", first_layer)
