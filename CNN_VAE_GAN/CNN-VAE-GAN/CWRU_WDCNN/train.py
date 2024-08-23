import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt  # 从matplotlib库中导入pyplot模块，用于绘图
import os
from torch import nn, optim  # 从torch库中导入nn模块（包含神经网络层和损失函数）和optim模块（包含优化算法）
# DataLoader（用于加载数据），dataset（数据集基类），random_split（用于随机分割数据集）
from torch.utils.data import DataLoader, dataset, random_split
import seaborn as sns  # 导入seaborn库，用于数据可视化
from fault_diag_utils import *  # 从fault_diag_utils模块导入所有工具函数，包含自定义的数据处理函数
from cnn_model import CNN as model
# from cnn_model import LSTM_CNN as model
sns.set()  # 设置seaborn的默认可视化风格


def train(net: model, dataloader: DataLoader, loss_func, optimizer: optim.Adam, epoch: int):
    '''训练CNN模型'''
    global DEVICE, LOSSES
    # 训练模式
    net.train()
    # 初始化平均损失变量为 0
    loss_mean = 0
    cnt = 0
    for i, (x, label) in enumerate(dataloader):
        # 将数据和标签移动到指定的设备上
        x, label = x.to(DEVICE), label.to(DEVICE)
        # 清空优化器梯度
        optimizer.zero_grad()
        # 计算预测值
        y = net(x).to(DEVICE)
        # 计算损失
        loss = loss_func(y, label)
        # 误差反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 统计平均损失
        loss_mean += loss.item()
        # 根据预测概率获取预测类别
        predict = y.max(1, keepdim=True)[1]
        # 计算正确预测的数量
        cnt += predict.eq(label.view_as(predict)).sum().item()

    # # 计算平均损失
    # avg_loss = loss_mean / len(dataloader)
    # # 计算准确率
    # acc = 100 * cnt / len(dataloader.dataset)
    # # 将平均损失添加到全局变量LOSSES字典中对应的键
    # LOSSES['train'].append(avg_loss)
    # # 将准确率添加到全局变量ACCS字典中对应的键
    # ACCS['train'].append(acc)


def test(net: model, dataloader: DataLoader, loss_func, datatype: str):
    '''测试模型的准确率'''
    global DEVICE, ACCS, LOSSES
    # 测试模式
    net.eval()
    # 初始化测试损失和正确预测计数
    test_loss = 0
    cnt = 0
    # 使用上下文管理器，关闭梯度计算
    with torch.no_grad():
        for x, label in dataloader:
            # 将数据和标签移动到指定的设备上
            x, label = x.to(DEVICE), label.to(DEVICE)
            # 前向传播，计算预测值
            y = net(x).to(DEVICE)
            # 计算损失
            loss = loss_func(y, label)
            # 统计平均损失
            test_loss += loss.item()
            # 根据预测概率获取预测类别
            predict = y.max(1, keepdim=True)[1]
            # 计算正确预测的数量
            cnt += predict.eq(label.view_as(predict)).sum().item()

    # 计算平均损失
    avg_loss = test_loss / len(dataloader)
    # 计算准确率
    acc = 100 * cnt / len(dataloader.dataset)
    # 将平均损失添加到全局变量LOSSES字典中对应的键
    LOSSES[datatype].append(avg_loss)
    # 将准确率添加到全局变量ACCS字典中对应的键
    ACCS[datatype].append(acc)
    # 打印测试数据的损失和准确率
    print("{} data:  average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)".format(datatype,
                                                                             avg_loss, cnt, len(dataloader.dataset),
                                                                             acc))


if __name__ == "__main__":
    '''定义相关超参数'''
    global DEVICE, BATCH_SIZE, EPOCHS, LENGTH, PATH, LOSSES, ACCS
    # 样本长度
    LENGTH = 1024
    # 批尺寸
    BATCH_SIZE = 128
    # 训练迭代次数
    EPOCHS = 80
    # 重叠采样的偏移量
    STRIDE = 100
    # 类别数量
    num_classes = 2
    PATH = "data1"
    # 定义训练设备这里使用GPU进行加速
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化损失和准确率字典
    LOSSES = {'train': [], 'test': []}
    ACCS = {'train': [], 'test': []}
    print(f"DEVICE:{DEVICE}")
    # 读取数据
    files = os.listdir(PATH)
    label = []
    data = []
    files_num = len(files) - 1
    for file in files:
        if "csv" not in file:
            continue
        labeli, xi = file_read(PATH + "/" + file)
        if labeli == 0:
            # 重叠采样
            for j in range(0, len(xi)-LENGTH, STRIDE):
                label.append(labeli)
                data.append(xi[j:j+LENGTH, :].T)
            # 额外截取最后一段数据
            label.append(labeli)
            data.append(xi[-LENGTH:, :].T)
        else:
            # 重叠采样
            for j in range(0, int(len(xi) / files_num - LENGTH), STRIDE):
                label.append(1)
                data.append(xi[j:j + LENGTH, :].T)
            # 额外截取最后一段数据
            label.append(1)
            data.append(xi[-LENGTH:, :].T)
    # 定义数据集
    ds = Data_set(data, label)
    '''数据集分割'''
    train_size = int(0.7*len(ds))  # 训练集的大小
    test_size = len(ds) - train_size  # 测试集的大小
    # 使用random_split随机分割数据集为训练集和测试集
    train_loader, test_loader = random_split(ds, [train_size, test_size])
    # 创建训练数据加载器
    train_loader = DataLoader(train_loader, BATCH_SIZE, shuffle=True)
    # 创建测试数据加载器
    test_loader = DataLoader(test_loader, BATCH_SIZE, shuffle=False)
    '''定义网络模型、优化器和损失函数'''
    net = model(DEVICE, out_channel=num_classes).to(DEVICE)
    optimizer = optim.Adam(net.parameters())
    loss_func = nn.CrossEntropyLoss()  # 损失函数使用交叉熵函数

    # 开始训练循环
    for epoch in range(EPOCHS):
        train(net, train_loader, loss_func, optimizer, epoch)
        # 每1个epoch进行一次测试
        if (epoch+1) % 1 == 0:
            print()
            test(net,  train_loader, loss_func, "train")
            test(net, test_loader, loss_func, "test")
            print()
    # 保存模型
    torch.save(net, "CWRU_cnn_net1.pth")
    # torch.save(net, "./cnn_lstm_net1.pth")
    # 训练完成后，导出模型为ONNX格式
    # 导出模型为ONNX格式
    # batch_size = 1  # 批尺寸
    # input_feature_size = 1  # 输入特征维度
    # sequence_length = LENGTH  # 序列长度
    # dummy_input = torch.randn(batch_size, input_feature_size, sequence_length).to(DEVICE)  # 创建一个与模型输入维度相同的dummy input
    # torch.onnx.export(net, dummy_input, "./cnn_lstm_net.onnx",
    #                   input_names=['input'], output_names=['output'],
    #                   dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    # print("模型已导出为ONNX格式")
    '''绘图部分'''
    # 训练完成后，绘制并保存图形
    plt.figure(figsize=(12, 6))  # 设置图形的大小
    # 绘制训练和测试的loss
    plt.subplot(1, 2, 1)  # 创建子图1/1列2行第1个
    plt.plot(range(1, EPOCHS + 1), LOSSES['train'], label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), LOSSES['test'], label='Test Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练和测试的准确率
    plt.subplot(1, 2, 2)  # 创建子图1/1列2行第2个
    plt.plot(range(1, EPOCHS + 1), ACCS['train'], label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), ACCS['test'], label='Test Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 保存图形
    plt.tight_layout()  # 调整子图布局以适应图形
    plt.savefig('CWRU_cnn_training_test_results.png')  # 保存为PNG文件
    # plt.savefig('cnnLstm_training_test_results.png')  # 保存为PNG文件
    # 显示图形
    plt.show()
    # # nvitop     ## 监控NVIDIA GPU
    # netron cnn_net.pth  #生成cnn_net网络结构图
    # netron cnn_lstm_net.pth  #生成cnn_lstm_net网络结构图
