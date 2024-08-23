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
# from sklearn.manifold import TSNE
# import seaborn as sns
from Cnnmodel import CNN2 as CNN


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
            num_file_to_read = len(os.listdir(root_dir)) - 1
            folder_path = os.path.join(root_dir, label)
            if os.path.isdir(folder_path):
                # 遍历文件夹内的图像文件
                if label == '0':
                    i = 0
                    for img_file in os.listdir(folder_path):
                        # # 只读取每个folder_path中的前200张图片
                        # i = i + 1
                        # if i > 200:
                        #     break
                        img_path = os.path.join(folder_path, img_file)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            # 将文件夹名称作为标签
                            self.labels.append(self.label_to_idx[label])
                else:
                    i = 0
                    img_files = os.listdir(folder_path)
                    num_imgs_to_read =int(len(img_files)/num_file_to_read)
                    for img_file in os.listdir(folder_path):
                        # 只读取每个folder_path中的前num_imgs_to_read张图片
                        i = i + 1
                        if i > num_imgs_to_read:
                            break
                        img_path = os.path.join(folder_path, img_file)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            # 将文件夹名称作为标签
                            self.labels.append(1)



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
    # 故障类别数
    num_classes = 2
    # 批尺寸
    BATCH_SIZE = 32
    # 训练迭代次数
    EPOCHS = 30
    # 定义训练设备这里使用GPU进行加速
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 保存损失以及准确率信息
    LOSSES = {'train': [], 'test': []}
    ACCS = {'train': [], 'test': []}
    print(f"DEVICE:{DEVICE}")
    # 创建数据集
    root_dir = './data2'
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((125, 187)),  # 根据需要调整图像大小
        transforms.ToTensor(),  # 将PIL图像转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    dataset = CustomImageDataset(root_dir=root_dir, transform=transform)

    # 划分数据集为训练集和测试集
    from torch.utils.data import random_split

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # # 展示一些图片
    # fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    # sample_idxs = np.random.choice(range(len(dataset)), 20)
    # for i, idx in enumerate(sample_idxs):
    #     img, _ = dataset[idx]
    #     axes[i // 5][i % 5].imshow(np.transpose(img.numpy(), (1, 2, 0)))  # 转换通道顺序以适应matplotlib
    #     axes[i // 5][i % 5].set_title(f'Sample {i + 1}')
    #     axes[i // 5][i % 5].axis('off')  # 不显示坐标轴
    # plt.tight_layout()
    # plt.show()

    # 将模型移动到DEVICE上
    model = CNN(num_classes, input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 优化器
    loss_func = nn.CrossEntropyLoss()  # 损失函数

    # 训练模型
    for epoch in range(EPOCHS):
        model.train()  # 设置模型为训练模式
        total_train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            outputs = model(images)
            loss = loss_func(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            # 累积损失和计算准确率
            total_train_loss += loss.item()
            predicted = outputs.max(1, keepdim=True)[1]
            # _, predicted = torch.max(outputs.data, 1, keepdim=True)
            # correct_train_preds += (predicted == labels).sum().item()
            correct_train_preds += predicted.eq(labels.view_as(predicted)).sum().item()
            total_train_samples += labels.size(0)

        # 计算平均损失和准确率
        average_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train_preds / total_train_samples
        LOSSES['train'].append(average_train_loss)
        ACCS['train'].append(train_accuracy)

        # 测试模型
        model.eval()  # 设置模型为评估模式
        total_test_loss = 0
        correct_test_preds = 0
        total_test_samples = 0

        with torch.no_grad():  # 在评估模式中不计算梯度
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = loss_func(outputs, labels)

                total_test_loss += loss.item()
                predicted = outputs.max(1, keepdim=True)[1]
                # _, predicted = torch.max(outputs.data, 1)
                # correct_test_preds += (predicted == labels).sum().item()
                correct_test_preds += predicted.eq(labels.view_as(predicted)).sum().item()
                total_test_samples += labels.size(0)

        average_test_loss = total_test_loss / len(test_loader)
        test_accuracy = correct_test_preds / total_test_samples
        LOSSES['test'].append(average_test_loss)
        ACCS['test'].append(test_accuracy)

        print(f"Epoch {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {average_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {average_test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}")

    # 可视化训练和测试的损失和准确率
    for split in ['train', 'test']:
        plt.plot(range(1, EPOCHS + 1), LOSSES[split], label=f'{split} loss')
        plt.plot(range(1, EPOCHS + 1), ACCS[split], label=f'{split} accuracy')
    plt.legend()
    plt.title('Training and Testing Loss and Accuracy')  # 可以添加一个标题
    plt.xlabel('Epochs')  # 可以添加x轴标签
    plt.ylabel('Loss / Accuracy')  # 可以添加y轴标签
    plt.grid(True)  # 可以添加网格线
    # 保存图形
    plt.savefig('loss_accuracy_plot2.png')  # 保存为PNG文件
    # 显示图形
    plt.show()

    # # 保存模型
    # # 定义基础文件名和扩展名
    # base_filename = 'gearRackDiagnosisNet'
    # extension = '.pth'
    # # 检查基础文件名加扩展名的文件是否存在
    # full_filename = base_filename + extension
    # if os.path.exists(full_filename):
    #     # 文件存在，找到所有匹配的文件
    #     files = [f for f in os.listdir('.') if f.startswith(base_filename) and f.endswith(extension)]
    #     # 从找到的文件名中获取最大的数字
    #     max_number = 0
    #     for file in files:
    #         try:
    #             # 尝试将文件名中的数字部分转换为整数
    #             number = int(''.join(filter(str.isdigit, file)))
    #             max_number = max(max_number, number)
    #         except ValueError:
    #             # 如果转换失败，忽略这个文件
    #             continue
    #
    #     # 增加数字
    #     max_number += 1
    #     # 使用新数字创建新的文件名
    #     full_filename = f"{base_filename}{max_number}{extension}"
    #     # print(f"文件已存在，新文件名将为：{full_filename}")
    # # 保存模型，使用新的文件名
    # torch.save(model.state_dict(), full_filename)

    # 保存模型
    torch.save(model.state_dict(), 'CWRUDiagnosisNet2.pth')

    # t-SNE聚类可视化
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
    #     activations, labels = get_layer_activations(model, train_loader, DEVICE, layer)
    #     tsne_results = TSNE(n_components=2, method='exact', metric='euclidean').fit_transform(activations)
    #     sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, ax=axes[i], palette="deep")
    #     axes[i].set_title(f't-SNE of {layer}')
    #     axes[i].legend(loc='best')
    #
    # plt.tight_layout()
    # plt.savefig('cnn2D_T-SNE聚类.png')  # 保存为PNG文件
    # plt.show()
    # # netron gearRackDiagnosisNet.pth  #生成网络结构图

