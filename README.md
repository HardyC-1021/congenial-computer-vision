# congenial-computer-vision
Some Operations of Computer vision.
一些常见的计算机视觉操作。

exe1 图像去燥、直方图均衡化。对图像的基本去燥功能，及应用直方图均衡化实现对图像的增强处理。

exe2 实现边缘检测、Hough变换。Canny边缘检测算子实现对图像的边缘检测；在边缘检测的基础上，应用Hough变换算法提取最可能存在于图像上的3条直线，给出直线表达式。

exe3 实现基于BoW特征的物体识别，学习物体识别过程。选用了汽车公开图像数据集，实现基于BoW特征的物体识别，在测试图像上的识别精度不低于70%。

exe4 自选数据集，实现基于LDA的人脸图像识别，在测试图像上的识别精度不低于80%。

exe5 在MNIST手写数字图像数据集上应用卷积神经网络进行数字识别，有完整训练和测试过程。选用的是MobileNetV3_small神经网络，网络模型再model.py文件中。

exe6 设计实现了一个综合图像处理系统。要求有界面UI，实现涉及的全部图像处理功能，包括：图像处理基本功能；边缘检测；图像特征提取；主成分分析；图像分割；物体检测；物体识别；卷积神经网络等。综合系统采用UI界面是Tkinter。
![image](https://user-images.githubusercontent.com/50348745/209603216-05dcbc56-a132-4daf-8d2a-1cd51155ddb9.png)
