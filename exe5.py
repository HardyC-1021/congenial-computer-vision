# 实验目的：学习卷积神经网络应用
# 实验要求：在MNIST手写数字图像数据集上应用卷积神经网络进行数字识别，要求有完整训练和测试过程。
# # Author: HardyC-1021
import torch
from torchvision import transforms  # 对图像进行原始的数据处理的工具
from torchvision import datasets  # 获取数据
from torch.utils.data import DataLoader  # 加载数据
from model import MobileNetV3_small
import datetime
import torch.optim as optim  # 与优化器有关
import torch.nn as nn
from PIL import Image
import cv2

# prepare dataset
batch_size = 64
# GPU对2的幂次的batch可以发挥更佳的性能，因此设置成16、32、64、128…时往往要比设置为整10、整100的倍数时表现更优
# 在神经网络训练时，常常需要采用批输入数据的方法，为此需要设定每次输入的批数据大小batch_size
transform = transforms.Compose([  # 处理图像
    transforms.ToTensor(),  # Convert the PIL Image to Tensor
    transforms.Resize((224,224)), # 归一化；0.1307为均值，0.3081为标准差
])

train_dataset = datasets.MNIST(root='./MNIST/', train=True, download=True, transform=transform)
# download=True表示自动下载MNIST数据集(建议科学上网，不然速度很慢，而且可能下载中断)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./MNIST/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


def train(epochs=5):
    time2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time2)
    net = MobileNetV3_small(num_classes=10)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            # 获得一个批次的数据和标签
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            # 获得模型预测结果(64, 10)
            outputs = net(inputs)
            # 交叉熵代价函数outputs(64,10),target（64）
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 300 == 299:  # batch_idx最大值为937；937*64=59968 意味着丢弃了部分的样本
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
                # 注：在python中，通过使用%，实现格式化字符串的目的；%d 有符号整数(十进制)
                running_loss = 0.0
    weight_path = 'Small-Mnist-cpu.pkl'
    torch.save(net.state_dict(), weight_path)
    time3 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time3)

def test():

    weight_path = 'Small-Mnist.pkl'

    net = MobileNetV3_small(num_classes=10)

    net.eval()
    net.load_state_dict(torch.load(weight_path))
    correct = 0  # 正确预测的数量
    total = 0  # 总数量
    with torch.no_grad():  # 测试的时候不需要计算梯度（避免产生计算图）
        for data in test_loader:
            inputs, labels = data[0], data[1]
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))

def detect(pic_path):

    pic = cv2.imread(pic_path,cv2.IMREAD_GRAYSCALE)
    img_tensor = transform(pic).unsqueeze(0)
    weight_path = 'Small-Mnist.pkl'
    net = MobileNetV3_small(num_classes=10)
    net.eval()
    net.load_state_dict(torch.load(weight_path))

    net_output = net(img_tensor)
    print(net_output)
    _, predicted = torch.max(net_output.data, 1)
    result = predicted[0].item()

    num = 'The detected number is: '+ str(result+1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(pic, num, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return pic
    # cv2.imshow('pic',pic)
    # cv2.waitKey(0)
if __name__=="__main__":
    # train(3)
    detect('5.jpg')
