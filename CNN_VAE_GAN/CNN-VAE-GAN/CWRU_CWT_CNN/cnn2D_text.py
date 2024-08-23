from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd



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


# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 存储所有图像的路径和标签
        self.images = []
        self.labels = []
        self.transform = transform
        self.root_dir = root_dir
        self.label_to_idx = {label: idx for idx, label in enumerate(os.listdir(root_dir))}

        # 遍历所有子文件夹
        for label in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, label)
            if os.path.isdir(folder_path):
                # 遍历文件夹内的所有图像文件
                for img_file in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_file)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        # 将文件夹名称作为标签
                        self.labels.append(self.label_to_idx[label])

        # # 遍历所有子文件夹
        # for label in os.listdir(root_dir):
        #     num_file_to_read = len(os.listdir(root_dir)) - 1
        #     folder_path = os.path.join(root_dir, label)
        #     if os.path.isdir(folder_path):
        #         # 遍历文件夹内的图像文件
        #         if label == '0':
        #             i = 0
        #             for img_file in os.listdir(folder_path):
        #                 # # 只读取每个folder_path中的前200张图片
        #                 # i = i + 1
        #                 # if i > 200:
        #                 #     break
        #                 img_path = os.path.join(folder_path, img_file)
        #                 if os.path.isfile(img_path):
        #                     self.images.append(img_path)
        #                     # 将文件夹名称作为标签
        #                     self.labels.append(self.label_to_idx[label])
        #         else:
        #             i = 0
        #             img_files = os.listdir(folder_path)
        #             num_imgs_to_read = int(len(img_files) / num_file_to_read)
        #             for img_file in os.listdir(folder_path):
        #                 # 只读取每个folder_path中的前num_imgs_to_read张图片
        #                 i = i + 1
        #                 if i > num_imgs_to_read:
        #                     break
        #                 img_path = os.path.join(folder_path, img_file)
        #                 if os.path.isfile(img_path):
        #                     self.images.append(img_path)
        #                     # 将文件夹名称作为标签
        #                     self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引获取图像路径
        img_path = self.images[idx]
        # 根据索引获取标签
        label = self.labels[idx]
        # 读取图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 返回图像和对应的标签
        return image, torch.tensor(label)

if __name__ == '__main__':
    # 图像尺寸 (channels, height, width)
    input_size = (3, 125, 187)
    # 类别数
    num_classes = 4
    # 批尺寸
    BATCH_SIZE = 32
    # 定义训练设备这里使用GPU进行加速
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE:{DEVICE}")
    # 创建数据集
    root_dir = './data1'
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((125, 187)),  # 根据需要调整图像大小
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    dataset = CustomImageDataset(root_dir=root_dir, transform=transform)

    # 创建数据加载器
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载保存的模型权重
    model = CNN2(num_classes, input_size)
    model.load_state_dict(torch.load('JNDiagnosisNet1.pth'))
    model.to(DEVICE)
    model.eval()

    # 测试模型
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(DEVICE), label.to(DEVICE)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算准确率
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Accuracy of the model on the test images: {accuracy}%')

    # 绘制混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=[str(i) for i in range(num_classes)],
                                  index=[str(i) for i in range(num_classes)])

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
    sns.heatmap(pd.DataFrame(accuracy_matrix), annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax1, xticklabels=False, yticklabels=False)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_title('Accuracy')

    # 错误率矩阵
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(pd.DataFrame(error_matrix), annot=True, fmt='.2%', cmap='Reds', cbar=False, ax=ax2, xticklabels=False, yticklabels=False)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    ax2.set_title('Error')

    # 保存混淆矩阵图形
    plt.savefig('confusion_matrix2.png')
    plt.show()

    # # t-SNE聚类可视化
    # def get_layer_activations(model, loader, device, layer):
    #     model.eval()
    #     activations = []
    #     labels_list = []
    #     with torch.no_grad():
    #         for images, labels in loader:
    #             images = images.to(device)
    #             x = model._forward_features(images)
    #             if layer == 'maxpool1':
    #                 x = model.maxpool1(x)
    #             elif layer == 'maxpool2':
    #                 x = model.maxpool2(x)
    #             elif layer == 'maxpool3':
    #                 x = model.maxpool3(x)
    #             x = x.view(x.size(0), -1)
    #             activations.append(x.cpu().numpy())
    #             labels_list.append(labels.numpy())
    #     return np.concatenate(activations), np.concatenate(labels_list)
    #
    #
    # layers = ['maxpool1', 'maxpool2', 'maxpool3']
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #
    # for i, layer in enumerate(layers):
    #     activations, labels = get_layer_activations(model, test_loader, DEVICE, layer)
    #     tsne_results = TSNE(n_components=2, method='exact', metric='euclidean').fit_transform(activations)
    #     sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, ax=axes[i], palette="deep")
    #     axes[i].set_title(f't-SNE of {layer}')
    #     axes[i].legend(loc='best')
    #
    # plt.tight_layout()
    # plt.savefig('cnn2D_T-SNE聚类.png')  # 保存为PNG文件
    # plt.show()

    # netron gearRackDiagnosisNet.pth  #生成网络结构图
