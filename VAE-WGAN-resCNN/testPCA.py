import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.utils import make_grid
from sklearn.metrics import confusion_matrix
import os
import time
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
from dataloader import CustomImageDataset
from models import VAE_GAN, Discriminator
from utils import show_and_save, plot_loss, TopHalfCrop

if __name__ == '__main__':
    # 批次大小
    batch_size = 603
    # 数据集路径
    root_dir = './data1'
    # 模型保存的文件夹
    models_dir = 'models0'
    # 定义gamma参数，用于模型中的折扣因子或加权系数
    gamma = 15

    # # 把保存在txt文件中的阈值读取出来
    # with open('eta.txt', 'r') as f:
    #     eta = float(f.read())
    #     print("读取到的阈值:", eta)

    # 打开文件并读取内容
    with open('eta.txt', 'r') as f:
        content = f.read()
    # 去除可能的空白字符，并按逗号分割字符串
    eta_values = content.strip().split(',')
    # 分别将分割后的字符串转换为浮点数或整数，并赋值给新变量
    # 这里假设阈值是浮点数，如果它们是整数，使用 int() 替换 float()
    eta_l = float(eta_values[0].strip())
    eta_h = float(eta_values[1].strip())
    # 打印读取的值以验证
    print("读取的阈值下限:", eta_l)
    print("读取的阈值上限:", eta_h)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((150, 100)),  # 根据需要调整图像大小
        TopHalfCrop(),  # 保留上半部分，裁剪掉下半部分
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    test_dataset = CustomImageDataset(root_dir=root_dir, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    real_batch = next(iter(test_loader))

    # 初始化模型结构
    # 创建VAE_GAN实例，以及单独的编码器、解码器和判别器实例
    vae_gan = VAE_GAN().to(device)


    # 加载保存的状态字典
    encoder_final_path = os.path.join(models_dir, 'encoder_final.pth')
    decoder_final_path = os.path.join(models_dir, 'decoder_final.pth')
    discriminator_final_path = os.path.join(models_dir, 'discriminator_final.pth')


    # 加载整个模型
    vae_gan_final_path = os.path.join(models_dir, 'vae_gan_final.pth')
    vae_gan_checkpoint = torch.load(vae_gan_final_path, map_location=device)
    vae_gan.encoder.load_state_dict(vae_gan_checkpoint['encoder_state_dict'])
    vae_gan.decoder.load_state_dict(vae_gan_checkpoint['decoder_state_dict'])
    vae_gan.discriminator.load_state_dict(vae_gan_checkpoint['discriminator_state_dict'])
    print('load model success')
    # 确保在加载模型后将其移动到GPU上
    vae_gan = vae_gan.to(device)
    discrim = Discriminator().to(device)

    # 定义损失函数
    criterion = nn.BCELoss().to(device)
    # 将模型设置为评估模式
    vae_gan.eval()
    discrim.eval()

    # 预先准备随机噪声z_fixed和真实样本x_fixed用于后续固定条件下的模型测试
    z_fixed = Variable(torch.randn((batch_size, 128)).to(device))
    x_fixed = Variable(real_batch[0].to(device))
    # 从生成模型中获取特定输出，用于后续操作
    b = vae_gan(x_fixed)[2]
    # 从计算图中分离张量，避免梯度计算
    b = b.detach()
    # 使用固定的随机向量通过生成模型的解码器，获取特定输出
    c = vae_gan.decoder(z_fixed)
    # 从计算图中分离张量，避免梯度计算
    c = c.detach()
    # 使用make_grid函数将特定输出转换为网格形式，并保存为图片
    show_and_save(f'test_noise', make_grid((c * 0.5 + 0.5).cpu(), 1))
    show_and_save(f'test', make_grid((b * 0.5 + 0.5).cpu(), 1))

    # 重构损失列表
    recon_loss_list = []
    # 预测标签
    y = []
    # 真实标签
    all_labels = []
    # 正确率
    correct = 0
    with torch.no_grad():
        for i, (data, lable) in enumerate(test_loader, 0):
            # 获取批量大小，用于后续的标签和数据准备
            bs = data.size()[0]

            datav = Variable(data.to(device))
            lable = Variable(lable.to(device))
            all_labels.extend(lable.cpu().numpy())
            # 通过生成模型获得输入数据的均值、对数方差和重构输出
            mean, logvar, rec_enc = vae_gan(datav)


            # 通过鉴别器获取重构数据的隐藏特征
            x_l_tilda = discrim(rec_enc)[1]
            # 通过鉴别器获取原始数据的隐藏特征
            x_l = discrim(datav)[1]
            # 计算重构损失，即隐藏特征的平方差的均值
            rec_loss = ((x_l_tilda - x_l) ** 2)
            # # 计算解码器的错误，结合重构损失和生成对抗损失
            # err_dec = gamma * rec_loss - gan_loss
            # 将重构损失添加到列表中，用于后续统计或输出
            # recon_loss_list.append(rec_loss.mean().item())
            recon_loss_list.extend(rec_loss.cpu().numpy())

            # Calculate the anomaly score
            # Apply PCA to reduce the dimensionality
            pca = PCA(n_components=1)
            recon_loss_list = pca.fit_transform(recon_loss_list)
            # batch_score = np.mean(recon_loss_list)
            #
            # if (batch_score < eta_l) or (batch_score > eta_h):
            #     print("异常!")
            # else:
            #     print("正常")

            scores = recon_loss_list
            recon_loss_list = []

            # Compare the scores with η to determine anomalies
            y_batch = []
            for lab, score in zip(lable, scores):
                anomalies = (score < eta_l) or (score > eta_h)
                if anomalies:
                    a = 0
                    # print("异常!")
                else:
                    a = 1
                    # print("正常")
                y.append(a)
                y_batch.append(a)

                a_tensor = torch.tensor(a)
                # lable_tensor = torch.tensor(lable)

                correct += (a_tensor.float() == lab.float()).sum().item()
            y_batch = sum(y_batch)
            if y_batch / batch_size >= 0.9 :
                print("正常")
            else:
                print("异常！")



    # 计算准确率
    y = np.array(y)
    accuracy = 100 * correct / len(all_labels)
    print(f'Accuracy of the model on the test images: {accuracy}%')

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(all_labels, y)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[str(i) for i in range(2)],
                                  index=[str(i) for i in range(2)])

    # 计算正确率和错误率矩阵
    accuracy_matrix = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    error_matrix = 1 - accuracy_matrix

    # 将正确率和错误率转换为一维数组
    accuracy_array = accuracy_matrix.ravel()
    error_array = error_matrix.ravel()

    # 创建正确率和错误率的DataFrame
    accuracy_matrix_df = pd.DataFrame([accuracy_array], columns=conf_matrix_df.columns)
    error_matrix_df = pd.DataFrame([error_array], columns=conf_matrix_df.columns)

    # 绘制混淆矩阵和正确率/错误率矩阵
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1, 1], wspace=0.3, hspace=0.3)

    # 混淆矩阵
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax0)
    ax0.set_xlabel('Predicted')
    ax0.set_ylabel('True')

    # 正确率矩阵
    ax1 = fig.add_subplot(gs[0, 1])
    sns.heatmap(pd.DataFrame(accuracy_matrix), annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=False, yticklabels=False)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_title('Accuracy')

    # 错误率矩阵
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(pd.DataFrame(error_matrix), annot=True, fmt='.2%', cmap='Reds', cbar=False, ax=ax2,
                xticklabels=False, yticklabels=False)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    ax2.set_title('Error')

    # 保存混淆矩阵图形
    plt.savefig('models0_confusion_matrix.png')
    plt.show()


