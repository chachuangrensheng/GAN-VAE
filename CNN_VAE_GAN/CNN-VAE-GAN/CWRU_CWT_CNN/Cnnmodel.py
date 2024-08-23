# -*- coding:UTF-8 -*- #

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN1, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)  # 使用inplace=True减少内存使用
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=True)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 64)
        # self.dropout1 = nn.Dropout(0.5)  # 添加dropout层
        self.fc2 = nn.Linear(64, 32)
        # self.dropout2 = nn.Dropout(0.5)  # 添加dropout层
        self.fc3 = nn.Linear(32, num_classes)

        self.classifier = nn.LogSoftmax(dim=1)  # 用于分类的log-softmax

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        # x = self.relu3(self.bn3(self.conv3(x)))
        # x = self.maxpool3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)  # 添加dropout层
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)  # 添加dropout层
        x = self.fc3(x)
        x = self.classifier(x)
        return x


class CNN2(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN2, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)  # 使用inplace=True减少内存使用
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        # self.dropout1 = nn.Dropout(0.5)  # 添加dropout层
        self.fc2 = nn.Linear(128, 64)
        # self.dropout2 = nn.Dropout(0.5)  # 添加dropout层
        self.fc3 = nn.Linear(64, num_classes)

        self.classifier = nn.LogSoftmax(dim=1)  # 用于分类的log-softmax

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)  # 添加dropout层
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x)  # 添加dropout层
        x = self.fc3(x)
        x = self.classifier(x)
        return x


class CNN3(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN3, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 新增的卷积层4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 256)  # 根据新增层调整全连接层输入特征数量
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.classifier = nn.LogSoftmax(dim=1)

        # 在合适的位置添加Dropout层
        self.dropout1 = nn.Dropout(p=0.2)  # p是Dropout概率
        self.dropout2 = nn.Dropout(p=0.3)

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        # 通过新增的卷积层4
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.maxpool4(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)  # 添加dropout层
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # 添加dropout层
        x = self.fc3(x)
        x = self.classifier(x)
        return x

class CNN4(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN4, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)  # 使用inplace=True减少内存使用
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, num_classes)
        # self.dropout1 = nn.Dropout(0.5)  # 添加dropout层
        # self.fc2 = nn.Linear(128, num_classes)
        # self.dropout2 = nn.Dropout(0.5)  # 添加dropout层
        # self.fc3 = nn.Linear(64, num_classes)

        self.classifier = nn.LogSoftmax(dim=1)  # 用于分类的log-softmax

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)  # 添加dropout层
        # x = self.fc2(x)
        # x = self.dropout2(x)  # 添加dropout层
        # x = self.fc3(x)
        x = self.classifier(x)
        return x


class CNN5(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN5, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  # 假设将64维特征压缩到32维
        self.fc4 = nn.Linear(32, num_classes)  # 新增加的全连接层

        # 用于分类的log-softmax
        self.classifier = nn.LogSoftmax(dim=1)

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 应用新增加的全连接层
        x = self.fc4(x)  # 然后应用最终的分类全连接层
        x = self.classifier(x)  # 应用分类器
        return x

class CNN6(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN6, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 新增的卷积层4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 新增的卷积层5
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 512)  # 根据新增层调整全连接层输入特征数量
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.classifier = nn.LogSoftmax(dim=1)

        # 在合适的位置添加Dropout层
        self.dropout1 = nn.Dropout(p=0.2)  # p是Dropout概率
        self.dropout2 = nn.Dropout(p=0.3)

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        # 通过新增的卷积层4
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.maxpool4(x)
        # 通过新增的卷积层5
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.maxpool5(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        # x = self.dropout1(x)  # 添加dropout层
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # 添加dropout层
        x = self.fc3(x)
        x = self.classifier(x)
        return x

class CNN7(nn.Module):
    def __init__(self, num_classes, input_size):
        super(CNN7, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  # 将64维特征压缩到32维
        self.fc4 = nn.Linear(32, 16)  # 新增加的全连接层4
        self.fc5 = nn.Linear(16, num_classes)  # 新增加的全连接层5

        # 用于分类的log-softmax
        self.classifier = nn.LogSoftmax(dim=1)

    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # 应用新增加的全连接层4
        x = F.relu(self.fc4(x))  # 应用新增加的全连接层5
        x = self.fc5(x)  # 然后应用最终的分类全连接层
        x = self.classifier(x)  # 应用分类器
        return x