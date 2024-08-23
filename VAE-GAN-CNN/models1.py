import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# input_size = (3, 125, 187)
input_size = (3, 75, 100)
height=75
width=100
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calculate_output_size(input_height, input_width, num_conv_layers=3, kernel_size=3, padding=0, stride=1):

    # 遍历每个卷积层
    for _ in range(num_conv_layers):
        # 每次卷积后尺寸减少2
        input_height = (input_height+2*padding - kernel_size)//stride + 1
        input_width = (input_width+2*padding - kernel_size)//stride + 1

    # 返回最终尺寸
    return input_height, input_width

fc_height,fc_width = calculate_output_size(height, width, num_conv_layers=3, kernel_size=3, padding=1, stride=1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 定义网络结构
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32,momentum=0.9)
        self.relu1 = nn.LeakyReLU(0.2)  # 使用inplace=True减少内存使用

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64,momentum=0.9)
        self.relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128,momentum=0.9)
        self.relu3 = nn.LeakyReLU(0.2)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 1024)
        self.bn4 = nn.BatchNorm1d(1024, momentum=0.9)
        self.fc_mean = nn.Linear(1024, 128)
        self.fc_logvar = nn.Linear(1024, 128)  # latent dim=128


    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.relu3(self.bn4(self.fc1(x)))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        encoder = Encoder()
        # 调用Encoder实例的_get_conv_output函数
        self.conv_output_size = encoder._get_conv_output(input_size)
        self.fc = nn.Linear(128, self.conv_output_size)
        self.bn1 = nn.BatchNorm1d(self.conv_output_size, momentum=0.9)
        self.relu = nn.LeakyReLU(0.2)
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=0.9)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.relu(self.bn1(self.fc(x)))
        # x = x.view(-1, self.conv_output_size)
        x = x.view(-1, 128, fc_height,fc_width)
        x = self.relu(self.bn2(self.deconv1(x)))
        x = self.relu(self.bn3(self.deconv2(x)))
        x = self.relu(self.bn4(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.9)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)

        # 计算全连接层输入特征数量
        conv_output_size = self._get_conv_output(input_size)
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.9)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def _get_conv_output(self, shape):
        # 创建一个虚拟的输入张量，计算通过卷积层后的输出大小
        input = torch.rand(1, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn1(self.conv2(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv4(x)))
        return x
    def forward(self, x):
        batch_size = x.size()[0]
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x1 = x
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))

        return x, x1


class VAE_GAN(nn.Module):
    def __init__(self):
        super(VAE_GAN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.discriminator.apply(weights_init)

    def forward(self, x):
        bs = x.size()[0]
        z_mean, z_logvar = self.encoder(x)
        std = z_logvar.mul(0.5).exp_()

        # sampling epsilon from normal distribution
        epsilon = Variable(torch.randn(bs, 128).half().to(device))
        z = z_mean + std * epsilon
        x_tilda = self.decoder(z)

        return z_mean, z_logvar, x_tilda


