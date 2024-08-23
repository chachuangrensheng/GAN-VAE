import torch
from torch import nn


class CNN(nn.Module):
    '''定义一维卷积神经网络模型'''

    def __init__(self, DEVICE, in_channel=1, out_channel=2):
        super(CNN, self).__init__()
        '''除输入层外，每个层级都包含了卷积、激活和池化三层'''
        '''输出层额外包含了BatchNorm层，提高网络收敛速度以及稳定性'''
        '''第一层卷积核大小为64，之后逐层递减'''
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24), # 一维卷积层，输入通道数为in_channel，输出通道数为16，卷积核大小为64，步长为16，填充为24。
            nn.BatchNorm1d(16),  # 批量归一化层，用于提高网络的稳定性和收敛速度，输出通道数为16
            nn.ReLU(inplace=True),  # 激活层，使用ReLU函数，inplace=True表示在原地进行计算，减少内存使用
            nn.MaxPool1d(kernel_size=2, stride=2)   # 池化层，使用最大池化，池化窗口大小为2，步长为2
        )


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, padding=8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=8, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 全连接层定义 引入Dropout机制以提高泛化能力
        self.fc = nn.Sequential(
            nn.Linear(64, 128),  # 第一个全连接层，将64维的输入映射到128维
            nn.Dropout(0.3),  # Dropout层，丢弃率为0.3，用于防止过拟合，提高模型的泛化能力
            nn.ReLU(inplace=True),
            nn.Linear(128, out_channel)  # 第二个全连接层，将128维的输入映射到out_channel维
        )
        # 使用softmax函数以计算输出从属于每一类的概率，1表示Softmax函数应用于最后一个维度
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        '''前向传播'''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # 将输出x重塑为适合全连接层的形状，size(0)是批次大小，-1表示自动计算其他维度的大小
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), 1)  # 再次重塑输出x，这次是为了应用Softmax层
        x = self.softmax(x)
        return x


class LSTM_CNN(nn.Module):
    '''定义LSTM-CNN网络模型'''

    def __init__(self, DEVICE, in_channel=1, out_channel=10):
        super(LSTM_CNN, self).__init__()
        self.DEVICE = DEVICE
        '''LSTM相关神经元定义'''
        self.lstm_layer1 = nn.LSTM(in_channel, 32) # 定义第一个LSTM层，输入特征大小为in_channel，输出特征大小为32
        self.lstm_layer2 = nn.LSTM(64, 1)
        self.lstm_fc1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64)
        )

        self.lstm_fc2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

        '''CNN相关神经元定义'''
        self.cnn_layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.cnn_layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.cnn_layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel)
        )
        # 使用softmax函数以计算输出从属于每一类的概率
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        '''前向传播'''
        '''*******LSTM*******'''
        x_lstm = x.permute(0, 2, 1)  # 调整输入数据的维度，以适应LSTM层的输入要求
        x_lstm.to(self.DEVICE)
        # 初始化隐藏神经元
        h1 = torch.zeros(1, 128, 32).to(self.DEVICE)  # 初始化第一个LSTM层的隐藏状态
        c1 = torch.zeros_like(h1).to(self.DEVICE)  # 初始化第一个LSTM层的细胞状态
        h2 = torch.zeros(1, 128, 1).to(self.DEVICE)
        c2 = torch.zeros_like(h2).to(self.DEVICE)
        y_lstm_ = []
        # 对原时序信号分段
        for i in range(8):
            x_lstm_ = x_lstm[:, i*128:(i+1)*128]  # 从输入数据中取出一个128长度的段
            y, (h1, c1) = self.lstm_layer1(x_lstm_, (h1, c1))  # 将信号段输入第一个LSTM层，并更新隐藏状态和细胞状态
            y = self.lstm_fc1(y)  # 将LSTM层的输出通过全连接层
            y, (h2, c2) = self.lstm_layer2(y, (h2, c2))  # 将全连接层的输出作为第二个LSTM层的输入
            y.to(self.DEVICE)
            y_lstm_.append(y)
        # 合并每一段的结果
        y_lstm = torch.cat(y_lstm_, 1)  # 将LSTM层的所有输出在特征维度上进行拼接
        y_lstm = y_lstm.view(y_lstm.size(0), -1)  # 调整LSTM输出的维度，以适应全连接层
        y_lstm = self.lstm_fc2(y_lstm)  # 将拼接后的LSTM输出通过另一个全连接层
        '''*******通过定义的各层进行CNN的前向传播*******'''
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        x = self.cnn_layer3(x)
        x = self.cnn_layer4(x)
        x = self.cnn_layer5(x)
        x = x.view(x.size(0), -1)  # 调整CNN输出的维度，以适应全连接层
        '''******LSTM+CNN******'''
        # 连接LSTM和CNN的输出，并通过全连接神经元
        x = torch.cat([x, y_lstm], 1)  # 在特征维度上拼接CNN和LSTM的输出
        x = self.fc(x)  # 将拼接后的输出通过全连接层
        x = x.view(x.size(0), x.size(1), 1)  # 调整全连接层输出的维度，以适应Softmax层
        y = self.softmax(x)  # 将全连接层的输出通过Softmax层，得到概率分布
        return y
