from __future__ import division
from sklearn.metrics import confusion_matrix
import os
import time
from PIL import Image
from sklearn.decomposition import PCA
# import reverse_pianoroll
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ops import *
from utils import *
import gc

barTime = 2.  # 2sec (fixed)

stride = 1
filterH = 3
filterW = 3

FS = 8  # 1/FS=0.125 sec per pitch


# 计算使用相同填充时卷积操作后的输出尺寸
# 当卷积窗口和输入尺寸的步长相同时，此函数用于计算输出尺寸
# 参数:
#   size: 输入尺寸
#   stride: 步长
# 返回值:
#   输出尺寸
def conv_out_size_same(size, stride):
    # 使用向上取整来确保输出尺寸是整数
    # 因为卷积操作在边缘使用填充来保持尺寸，所以输出尺寸通常略小于输入尺寸
    return int(math.ceil(float(size) / float(stride)))


class VAEGAN(object):
    def __init__(self, sess, flags, input_height=88, input_width=int(barTime * FS), batch_size=8, sample_num=3,
                 output_height=88, output_width=int(barTime * FS),
                 z_dim=64, gf_dim=6, df_dim=6,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='Nottingham',
                 input_fname_pattern='*.png', checkpoint_dir='./checkpoint', sample_dir='samples'):

        """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for Z. [128]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [8]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [8]
      gfc_dim: (optional) Dimension of gen units for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of input.
    """

        self.Flags = flags
        self.sess = sess

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width  # int(barTime * FS) = 16
        self.output_height = output_height
        self.output_width = output_width  # int(barTime * FS) = 16

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.qf_dim = df_dim  # encoder

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        # 归一化
        self.d_bn1 = batch_norm(name='d_bn1')

        self.d_bn2 = batch_norm(name='d_bn2')

        self.q_bn0 = batch_norm(name='q_bn0')
        self.q_bn1 = batch_norm(name='q_bn1')
        self.q_bn2 = batch_norm(name='q_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.images = []
        self.labels = []
        # self.transform = transform
        self.root_dir = self.dataset_name
        self.label_to_idx = {label: idx for idx, label in enumerate(os.listdir(self.root_dir))}

        # 根据数据集加载图片数据
        self.data_X, self.data_Xp, self.data_test, self.data_y = self.load_images()

        # # 根据特定的数据集名称加载数据
        # self.data_X, self.data_Xp, self.data_y = self.load_Nottingham()
        # 确定输入数据的特征维度
        self.c_dim = c_dim

        self.build_model()

    def load_images(self):
        """
        加载图片数据集。
        """

        # image_files = [f for f in os.listdir('path/to/your/image/folder') if f.endswith(self.input_fname_pattern)]
        # images = []
        images, lables = self.load_image()  # 假设load_image是您的图片加载函数

        # 测试数据
        x_test = images
        y_test = lables

        # 训练数据
        X = images[1:]
        # 分离出数据集中的前一个图片，用于序列预测
        xp = images[:-1]
        # 设置随机种子，以确保数据打乱的结果可复现
        seed = 500  # random.randint(1,1000)
        # 生成随机索引
        num_images = len(self.images) - 1
        indices = np.arange(num_images)

        # 使用相同的种子来打乱索引
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(indices)

        # 使用随机索引来打乱 X 和 xp
        # X = X[shuffled_indices]
        # xp = xp[shuffled_indices]
        X = [X[i] for i in shuffled_indices]
        xp = [xp[i] for i in shuffled_indices]
        X = np.array(X)
        xp = np.array(xp)
        y_test = np.array(y_test)
        x_test = np.array(x_test)
        return X, xp, x_test, y_test

    def load_image(self):
        """
        加载图片函数。
        """
        # 遍历所有子文件夹
        for label in os.listdir(self.root_dir):
            num_file_to_read = len(os.listdir(self.root_dir)) - 1
            folder_path = os.path.join(self.root_dir, label)
            if os.path.isdir(folder_path):
                # 遍历文件夹内的图像文件
                if label == '0':
                    i = 0
                    for img_file in os.listdir(folder_path):
                        # # 只读取每个folder_path中的前200张图片
                        # i = i + 1
                        # if i > 2:
                        #     break
                        img_path = os.path.join(folder_path, img_file)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            # 将文件夹名称作为标签
                            self.labels.append(1)
                elif label != '0' and not self.Flags.train:
                    i = 0
                    img_files = os.listdir(folder_path)
                    num_imgs_to_read = int(len(img_files) / num_file_to_read)
                    for img_file in os.listdir(folder_path):
                        # 只读取每个folder_path中的前num_imgs_to_read张图片
                        i = i + 1
                        if i > num_imgs_to_read:
                            break
                        img_path = os.path.join(folder_path, img_file)
                        if os.path.isfile(img_path):
                            self.images.append(img_path)
                            # 将文件夹名称作为标签
                            self.labels.append(0)

        # 将图像列表转换为一个批次的列表
        images_batch = self._load_images_batch(self.images)

        # 返回图像批次和对应的标签列表
        return images_batch, self.labels

    def _load_images_batch(self, image_paths):
        """
        加载图像批次
        """
        # 使用列表推导式加载并预处理所有图像
        images_batch = [self._load_and_preprocess_image(img_path) for img_path in image_paths]
        # 将列表转换为一个批次的张量
        return images_batch

    def _load_and_preprocess_image(self, img_path):
        """
        加载和预处理单个图像。
        """
        cropped_image = imageio.imread(img_path).astype(np.float32)
        # 应用图像归一化
        normalized_image = np.array(cropped_image) / 127.5 - 1
        return normalized_image
        # 使用PIL加载图像
        # with Image.open(img_path) as img:
            # 将PIL图像转换为Tensor
            # img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            # print(f"Image tensor shape before check: {img_tensor.shape}")  # 打印原始图像张量的形状

            # img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            # img_tensor = tf.image.per_image_standardization(img_tensor)
            # with tf.compat.v1.Session() as sess:
            #     img_tensor = sess.run(img_tensor)

            # 返回预处理后的图像张量
            # return normalized_image

    def build_model(self):
        """
        构建VAE模型的图形计算图。包括定义输入占位符、编码器、解码器以及判别器的相关操作。
        这里不涉及具体的训练逻辑，只关注模型结构的搭建。
        """
        # 定义输入图像的维度
        input_dims = [self.input_height, self.input_width, self.c_dim]

        # 定义训练集输入图像的张量
        # encoder
        self.x0 = tf.Variable(tf.zeros([self.batch_size] + input_dims), dtype=tf.float32, name='real_inputsP')
        # 定义测试集输入图像的张量
        self.x0test = tf.Variable(tf.zeros([1] + input_dims), dtype=tf.float32, name='real_inputsPtest')
        # 定义训练集重构图像的张量
        self.x = tf.Variable(tf.zeros([self.batch_size] + input_dims), dtype=tf.float32, name='real_inputs')
        # 定义采样图像的张量
        # self.sample_x = tf.Variable(tf.zeros([self.sample_num] + input_dims), dtype=tf.float32, name='sample_x')

        # 编码器部分的操作
        x0 = self.x0
        x = self.x
        # sample_x = self.sample_x
        # 定义潜变量的张量
        self.zp = tf.Variable(tf.zeros([self.batch_size, self.z_dim]), dtype=tf.float32, name='zp')
        # self.zp1 = tf.Variable(tf.zeros([1, self.z_dim]), dtype=tf.float32, name='zp')

        # 通过编码器获取潜变量的均值和标准差
        # 构建编码器
        self.z_mean, self.z_log_sigma_sq = self.Encoder(x0)

        # 构建测试编码器
        self.z_mean1, self.z_log_sigma_sq1 = self.Encoder(self.x0test, reuse=True)

        # 从潜变量分布中采样
        eps = tf.random.normal((self.batch_size, self.z_dim), 0, 1, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
        # self.feature_op = self.z

        # 通过解码器重构原始输入
        # 构建解码器
        self.x0_tilde = self.generator(self.z)
        # 在测试阶段使用
        self.xp_tilde = self.generator(self.zp, reuse=True)

        # 判别器对原始输入和重构输入的判断
        # 构建判别器
        self.Dx, self.Dx_logits,  _ = self.discriminator(x)
        self.Dx0, self.Dx0_logits, _ = self.discriminator(x0, reuse=True)
        # self.DxT, self.DxT_logits, _ = self.discriminator(self.x0test, reuse=True, text=True)  # 在测试阶段使用
        self.Dx0_tilde, self.Dx0_logits_tilde, _ = self.discriminator(self.x0_tilde, reuse=True)
        self.Dxp_tilde, self.Dxp_logits_tilde, _ = self.discriminator(self.xp_tilde, reuse=True)

        self.feature_op = self.Dx_logits
        # self.feature_opT = self.DxT_logits
        # 定义用于从潜变量采样的操作
        # 定义采样器
        self.sampler = self.generator(self.zp, reuse=True)  # zp: will be filled later with random noise

        def sigmoid_cross_entropy_with_logits(x, y):
            """
            计算sigmoid交叉熵损失。

            该函数为逻辑回归模型的损失函数，适用于二分类问题。它首先将输入x应用sigmoid函数，
            然后计算与目标值y的交叉熵损失。sigmoid函数将输入映射到0到1的范围内，这使得
            损失函数对模型的预测能力敏感，特别是在分类边界附近。

            参数:
            x: logits，模型的未经过sigmoid激活的输出。
            y: 标签，表示每个样本的真实分类。

            返回值:
            返回TensorFlow操作，该操作计算并返回sigmoid交叉熵损失。
            """
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        # 变分自编码器（VAE）损失
        # 计算变分自编码器的隐变量损失（KL散度）
        # KL_loss / Lprior
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                                - tf.square(self.z_mean)
                                                - tf.exp(self.z_log_sigma_sq), axis=1)

        # 计算VAE的重构损失（学习到的相似性度量）
        # Lth 层损失 - “学习到的相似性度量”
        self.LL_loss = 0.5 * (
            tf.reduce_sum(tf.square(self.Dx_logits - self.Dx0_logits_tilde))  # / (self.input_width*self.input_height)
        )

        # 计算VAE的整体损失，按输入维度平均
        self.vae_loss = tf.reduce_mean(self.latent_loss + self.LL_loss) / (
                self.input_width * self.input_height * self.c_dim)

        # 生成对抗网络（GAN）损失

        # 计算判别器在真实数据输入时的损失
        self.d_loss_real = 0.5 * (
            tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dx_logits, tf.ones_like(self.Dx)))
        )

        # 计算判别器在由生成器产生的假数据输入时的损失
        self.d_loss_fake = 0.5 * (
                tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.Dx0_logits_tilde, tf.zeros_like(self.Dx0_tilde)))
                + tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.Dxp_logits_tilde, tf.zeros_like(self.Dxp_tilde)))
        )

        # 计算判别器的总损失
        self.d_loss = self.d_loss_real + self.d_loss_fake

        # 计算生成器的损失，目标是让判别器将生成的数据误认为是真实数据
        self.g_loss = (0.5 * (tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.Dx0_logits_tilde, tf.ones_like(self.Dx0_tilde)))
                              + tf.reduce_mean(
                    sigmoid_cross_entropy_with_logits(self.Dxp_logits_tilde, tf.ones_like(self.Dxp_tilde))))
                       + tf.reduce_mean(self.LL_loss / (self.input_width * self.input_height * self.c_dim)))

        # 根据变量名将模型的可训练变量分为几组：编码器（q），判别器（d），生成器（g），以及VAE（q + g）
        t_vars = tf.compat.v1.trainable_variables()

        self.q_vars = [var for var in t_vars if 'q_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.vae_vars = self.q_vars + self.g_vars

        # 创建一个保存器以保存和恢复模型参数
        self.saver = tf.compat.v1.train.Saver()

    def train(self, config):
        """
        训练模型的函数，包括VAE、判别器D和生成器G三个网络的训练。

        :param config: 配置文件，包含训练的参数。
        """
        # 定义学习率占位符
        lr_E = tf.compat.v1.placeholder(tf.float32, shape=[])
        lr_D = tf.compat.v1.placeholder(tf.float32, shape=[])
        lr_G = tf.compat.v1.placeholder(tf.float32, shape=[])

        # 定义优化器Adam
        vae_optim = tf.compat.v1.train.AdamOptimizer(lr_E, beta1=config.beta1) \
            .minimize(self.vae_loss, var_list=self.vae_vars)
        d_optim = tf.compat.v1.train.AdamOptimizer(lr_D, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.compat.v1.train.AdamOptimizer(lr_G, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        # # 定义优化器GDO
        # vae_optim = tf.compat.v1.train.GradientDescentOptimizer(lr_E) \
        #     .minimize(self.vae_loss, var_list=self.vae_vars)
        # d_optim = tf.compat.v1.train.GradientDescentOptimizer(lr_D) \
        #     .minimize(self.d_loss, var_list=self.d_vars)
        # g_optim = tf.compat.v1.train.GradientDescentOptimizer(lr_G) \
        #     .minimize(self.g_loss, var_list=self.g_vars)



        # 初始化变量
        try:
            tf.compat.v1.global_variables_initializer().run()
        except:
            tf.compat.v1.initialize_all_variables().run()

        # 尝试加载模型
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 开始训练循环
        # 初始化计数器和开始时间
        counter = 1
        start_time = time.time()

        for epoch in xrange(config.epoch):
            # 计算每批数据的数量
            batch_idxs = min(self.data_X.shape[0], config.train_size) // self.batch_size

            # 遍历每批数据
            for idx in xrange(0, batch_idxs):
                # 设置学习率
                g_current_lr = 0.0005
                d_current_lr = 0.0001
                e_current_lr = 0.0005

                # 生成随机的z向量和批处理输入
                batch_z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))
                batch_inputs = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_inputsP = self.data_Xp[idx * self.batch_size:(idx + 1) * self.batch_size]



                # 更新VAE网络
                for i in range(2):
                    _, summary_str = self.sess.run([vae_optim, self.vae_loss],
                                                   feed_dict={lr_E: e_current_lr, self.x: batch_inputs,
                                                              self.x0: batch_inputsP, self.zp: batch_z})
                    errVAE = self.vae_loss.eval({self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z})

                # 更新D网络
                _, summary_str = self.sess.run([d_optim, self.d_loss],
                                               feed_dict={lr_D: d_current_lr, self.x: batch_inputs,
                                                          self.x0: batch_inputsP, self.zp: batch_z})
                errD_fake = self.d_loss_fake.eval({self.x0: batch_inputsP, self.zp: batch_z})
                errD_real = self.d_loss_real.eval({self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z})
                errD = errD_fake + errD_real

                # 更新G网络
                for i in range(2):
                    _, summary_str = self.sess.run([g_optim, self.g_loss],
                                                   feed_dict={lr_G: g_current_lr, self.x: batch_inputs,
                                                              self.x0: batch_inputsP, self.zp: batch_z})
                    errG = self.g_loss.eval({self.x: batch_inputs, self.x0: batch_inputsP, self.zp: batch_z})

                # 更新计数器和打印训练信息
                counter += 1
                print("Learning rates: [E: %.8f] [D: %.8f] [G: %.8f]" % (e_current_lr, d_current_lr, g_current_lr))
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, vae_loss: %.8f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs, time.time() - start_time, errVAE, errD, errG))

                # 保存样本和模型
                # if counter % 1 == 0:
                #     self.generateSamples(sample_dir=config.sample_dir, epoch=epoch, idx=idx)
            # if counter % 10 == 0:
            # 保存模型
            self.save(config.checkpoint_dir, counter)

        # Extract features from the test data using the trained model
        features_list = []
        # 计算每批数据的数量
        batch_idxs = min(self.data_X.shape[0], config.train_size) // self.batch_size
        # 遍历每批数据
        for idx in xrange(0, batch_idxs):
            batch_inputs = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
            features = self.extract_features(batch_inputs)
            features_list.append(features)
        # Apply PCA to reduce the dimensionality
        # pca = PCA(n_components=1)
        if isinstance(features_list, list):
            features_list = np.array(features_list)  # Convert the list to a NumPy array
        # Reshape the features_list to 2D if it's 3D
        if features_list.ndim == 3:
            features_list = features_list.reshape(-1, features_list.shape[2])
        # fz = pca.fit_transform(features_list)

        # Calculate the anomaly score
        normal_scores = features_list  # Assuming the first half are normal data

        # Calculate the threshold η
        mean_score = np.mean(normal_scores)
        std_score = np.std(normal_scores)
        eta = mean_score + 3 * std_score

        eta_mean = np.mean(eta)
        print("计算得到的阈值:", eta_mean)
        # 把eta_mean保存到txt文件中
        with open('eta_mean.txt', 'w') as f:
            f.write(str(eta_mean))

    # def load(self, checkpoint_dir):
    #     # Load the trained model from the checkpoint directory
    #     saver = tf.compat.v1.train.Saver()
    #     ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(self.sess, ckpt.model_checkpoint_path)
    #         return True
    #     else:
    #         print("No checkpoint found.")
    #         return False

    def test(self, config):

        """
        根据配置信息进行测试。

        此方法的目的是根据传入的配置对象，生成测试样本。它体现了测试过程的配置驱动特性，
        通过不同的配置可以针对不同的测试场景进行自动化测试。

        参数:
        - config: 包含测试配置信息的对象。其中应包含一个属性 `sample_dir`，
                  指定样本生成的目标目录。
        """
        # 根据配置信息生成测试样本
        self.generateSamples(sample_dir=config.sample_dir)

    def generateSamples(self, sample_dir, epoch=0, idx=0):
        """
        生成音乐样本文件。

        此方法利用训练好的编码器（Encoder）来生成音乐样本。它从数据集中随机选取一段音乐片段作为起始，
        将其送入编码器以获取初始状态，然后迭代地生成音乐小节并将其拼接起来形成完整的音乐作品。

        参数:
        - sample_dir: 保存生成音乐样本的目录。
        - epoch: 当前的轮次编号，用于命名生成的音乐文件。
        - idx: 编号索引，用于命名生成的音乐文件。
        """
        # 使用测试数据初始化编码器，设置reuse为True以允许重复使用相同的变量。
        # Encoder = self.Encoder(self.x0test, reuse=True, batch_size=1, train=False)

        # 循环生成指定数量的音乐样本。
        for n in range(self.sample_num):
            # 随机选择音乐片段的起始索引。
            sampleIndex = random.randint(1, self.data_Xp.shape[0])
            # 提取选定的音乐片段。
            x0 = self.data_Xp[sampleIndex:sampleIndex + 1]
            # image = None
            # 检查提取的音乐片段是否非空。
            # if x0.shape[0] > 0:

            # 迭代生成音乐小节。
            # saving the generated midi files!
            # for i in range(5):
            # 检查当前音乐片段是否非空。
            if x0.shape[0] > 0:
                # 初始化一个全零的音乐小节。
                bar = np.zeros(shape=[128, self.input_width, self.c_dim])
                # 将x0的值裁剪到[0, 1]区间。
                x0 = np.clip(x0, 0, 1)  # force to be [0-1]


                # 运行编码器以获得潜在变量的均值和方差的对数。
                z_mean1 = self.sess.run(self.z_mean1, feed_dict={self.x0test: x0})
                z_log_sigma_sq1 = self.sess.run(self.z_log_sigma_sq1, feed_dict={self.x0test: x0})
                # 根据正态分布生成随机噪声。
                eps = np.random.normal(0, 1, size=(1, self.z_dim))
                # 结合均值和噪声以获得潜在变量样本。
                sample_z = z_mean1 + np.sqrt(np.exp(z_log_sigma_sq1)) * eps
                # 使用潜在变量样本生成下一音乐片段。
                sample = self.sess.run(self.sampler, feed_dict={self.zp: sample_z})

                # 将sample的值裁剪到[0, 1]区间。
                sample = np.clip(sample, 0, 1)  # force to be [0-1]
                # 将裁剪后的sample缩放至范围[0, 127]。
                bar[33:121, :, :] = sample[0, :, :, :] * 127.0  # [0-127]
                # 将数据转换为整数类型。
                bar = bar.astype(int)

                # 如果是第一个音乐小节，将其赋值给image；否则，将其拼接到image后面。
                # if image is None:
                image = bar
                # else:
                #     image = np.concatenate((image, bar), axis=1)

                # 如果生成的音乐非空，则将其保存到指定目录。
                if np.amax(image) > 0:  # ignore the empty midi samples
                    print("\n[Sample]\n")
                    # 将音乐数据转换为PrettyMIDI对象并保存为MIDI文件。
                    # des_midi = reverse_pianoroll.piano_roll_to_pretty_midi(image, fs=FS, program=0)
                    # des_midi.write(sample_dir + '/train_' + str(epoch) + '_' + str(idx) + '_s' + str(n) + '.mid')
                    # 首先，将numpy数组转换为PIL Image
                    image_pil = Image.fromarray(image)
                    # 调整图片大小，如果需要的话
                    # image_pil = image_pil.resize((new_width, new_height), Image.ANTIALIAS)
                    # 保存图片
                    image_filename = sample_dir + str(idx) + '_s' + str(n) + '.png'
                    image_pil.save(image_filename)

                    print(f"Image saved as {image_filename}")

    # def test(self, config):
    #
    #     batchsize = 2
    #     # 预测标签
    #     y_list = []
    #     # 正确率
    #     correct = 0
    #     # 初始化计数器
    #     counter = 0
    #
    #     # # 尝试加载模型
    #     # could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    #     # if could_load:
    #     #     counter = checkpoint_counter
    #     #     print(" [*] Load SUCCESS")
    #     # else:
    #     #     print(" [!] Load failed...")
    #
    #     # 把保存在txt文件中的阈值读取出来
    #     with open('eta_mean.txt', 'r') as f:
    #         eta_mean = float(f.read())
    #         print("读取到的阈值:", eta_mean)
    #
    #     # 开始测试
    #     # for epoch in xrange(config.epoch):
    #     # 计算批次的数量
    #     batch_idxs = min(self.data_test.shape[0], config.train_size) // batchsize
    #     fz_list = []
    #     # 遍历每批数据
    #     for idx in xrange(0, batch_idxs):
    #         counter += 1
    #         # 批处理输入
    #         batch_inputs = self.data_test[idx * batchsize:(idx + 1) * batchsize]
    #         # Extract features from the test data using the trained model
    #         features = self.extract_features(batch_inputs, batchsize)
    #
    #         # Apply PCA to reduce the dimensionality
    #         # pca = PCA(n_components=1)
    #         # fz = pca.fit_transform(features)
    #         fz_list.append(features)
    #
    #         # Calculate the anomaly score
    #         scores = features
    #
    #         # Compare the scores with η to determine anomalies
    #         anomalies = scores > eta_mean
    #         if  anomalies.any():
    #             y = 0
    #             for _ in range(2):
    #                 y_list.append(0)
    #             # print("异常!")
    #         else:
    #             y = 1
    #             for _ in range(2):
    #                 y_list.append(1)
    #             # print("正常")
    #
    #         correct += (y == self.data_y[idx * batchsize:(idx + 1) * batchsize]).sum().item()
    #         if counter == 32:
    #             counter = 0
    #             batch_score = np.mean(fz_list)
    #             if batch_score > eta_mean:
    #                 print("异常!")
    #             else:
    #                 print("正常")
    #
    #     # 计算准确率
    #     accuracy = 100 * correct / len(self.data_y)
    #     print(f'Accuracy of the model on the test images: {accuracy}%')
    #
    #     # 绘制混淆矩阵
    #     y_list = np.array(y_list)
    #     conf_matrix = confusion_matrix(self.data_y, y_list)
    #     conf_matrix_df = pd.DataFrame(conf_matrix, columns=[str(i) for i in range(2)],
    #                                   index=[str(i) for i in range(2)])
    #
    #     # 计算正确率和错误率矩阵
    #     accuracy_matrix = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    #     error_matrix = 1 - accuracy_matrix
    #
    #     # 将正确率和错误率转换为一维数组
    #     accuracy_array = accuracy_matrix.ravel()
    #     error_array = error_matrix.ravel()
    #
    #     # 创建正确率和错误率的DataFrame
    #     accuracy_matrix_df = pd.DataFrame([accuracy_array], columns=conf_matrix_df.columns)
    #     error_matrix_df = pd.DataFrame([error_array], columns=conf_matrix_df.columns)
    #
    #     # 绘制混淆矩阵和正确率/错误率矩阵
    #     fig = plt.figure(figsize=(12, 8))
    #     gs = fig.add_gridspec(1, 3, width_ratios=[4, 1, 1], wspace=0.3, hspace=0.3)
    #
    #     # 混淆矩阵
    #     ax0 = fig.add_subplot(gs[0, 0])
    #     sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax0)
    #     ax0.set_xlabel('Predicted')
    #     ax0.set_ylabel('True')
    #
    #     # 正确率矩阵
    #     ax1 = fig.add_subplot(gs[0, 1])
    #     sns.heatmap(pd.DataFrame(accuracy_matrix), annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax1,
    #                 xticklabels=False, yticklabels=False)
    #     ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    #     ax1.set_title('Accuracy')
    #
    #     # 错误率矩阵
    #     ax2 = fig.add_subplot(gs[0, 2])
    #     sns.heatmap(pd.DataFrame(error_matrix), annot=True, fmt='.2%', cmap='Reds', cbar=False, ax=ax2,
    #                 xticklabels=False, yticklabels=False)
    #     ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    #     ax2.set_title('Error')
    #
    #     # 保存混淆矩阵图形
    #     plt.savefig('confusion_matrix0.png')
    #     plt.show()

    def extract_features(self, data, batch_size=1):
        # Convert the data to a NumPy array before feeding it to the session.
        # if isinstance(data, np.ndarray):
        #     data = tf.convert_to_tensor(data)
        # data_np = self.sess.run(data)
        if self.Flags.train:
            return self.sess.run(self.feature_op, feed_dict={self.x0: data})
        # else:
            # Encoder = self.Encoder(self.x0test, reuse=True)
            # # z_mean1, z_log_sigma_sq1 = self.sess.run(Encoder, feed_dict={self.x0test: data_np})
            # z_mean1 = self.sess.run(self.z_mean1, feed_dict={self.x0test: data})
            # z_log_sigma_sq1 = self.sess.run(self.z_log_sigma_sq1, feed_dict={self.x0test: data})
            # eps = tf.random.normal((batch_size, self.z_dim), 0, 1, dtype=tf.float32)
            # return  self.sess.run(self.feature_opT, feed_dict={self.x0test: data})

    def Encoder(self, Xp, y=None, reuse=False, batch_size=64, train=True):
        """
        构建编码器网络。

        参数:
        Xp: 输入图像的placeholder，shape为[batch_size, input_height, input_width, c_dim]。
        y: 未使用，保留以匹配接口。
        reuse: 是否重用变量。
        batch_size: 批处理大小。
        train: 是否在训练模式下运行。

        返回:
        z_mean: 均值向量，shape为[z_dim]。
        z_log_sigma_sq: 对数方差向量，shape为[z_dim]。
        """
        # 定义编码器的变量作用域
        with tf.compat.v1.variable_scope("Encoder") as scope:
            # 如果reuse为True，则重用变量
            if reuse:
                scope.reuse_variables()

            # # 设置conv2d和conv2d_transpose层的默认参数
            # with arg_scope([layers.Conv2D, layers.Conv2DTranspose],
            #                activation_fn=tf.nn.elu,
            #                normalizer_fn=layers.BatchNormalization,
            #                normalizer_params={'scale': True}
            #                ):
            # 将输入图像重塑为合适的形状
            net = tf.reshape(Xp, [-1, self.input_height, self.input_width, self.c_dim])

            # 直接在每个层中设置默认参数，而不是使用arg_scope
            net = tf.keras.layers.Conv2D(8, 5, strides=2, padding='same', activation='elu')(net)
            net = tf.keras.layers.BatchNormalization(scale=True)(net, training=train)
            net = tf.keras.layers.Conv2D(16, 5, strides=2, padding='same', activation='elu')(net)
            net = tf.keras.layers.BatchNormalization(scale=True)(net, training=train)
            net = tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='elu')(net)
            net = tf.keras.layers.BatchNormalization(scale=True)(net, training=train)

            # 将卷积层输出展平为一维向量
            net = tf.keras.layers.Flatten()(net)

            # # 建立全连接层，输出潜在空间的均值和对数方差
            # z_mean = layers.Dense(net, self.z_dim, activation_fn=None)
            # z_log_sigma_sq = layers.Dense(net, self.z_dim, activation_fn=None)
            # 建立全连接层，输出潜在空间的均值和对数方差
            z_mean = tf.keras.layers.Dense(self.z_dim, activation=None)(net)
            z_log_sigma_sq = tf.keras.layers.Dense(self.z_dim, activation=None)(net)

            # 返回潜在空间的均值和对数方差
            return z_mean, z_log_sigma_sq

    def discriminator(self, X, y=None, reuse=False, text=False):
        """
        判别器网络的定义。

        该函数构建了一个用于图像生成对抗网络（GAN）的判别器模型。它接收输入图像X，并判断其真实性。
        判别器通过一系列卷积和线性层进行处理，最终输出一个 sigmoid 激活值，表示输入图像的真实程度。

        参数:
        - X: 输入的图像数据，shape 为 [batch_size, height, width, channels]。
        - y: (可选) 输入的标签数据，本函数中未使用。
        - reuse: (布尔值) 指示是否重用变量。在构建多个判别器实例时使用。

        返回:
        - sigmoid_output: 判别器输出的 sigmoid 激活值，表示输入图像的真实性。
        - linear_output: 判别器输出的线性值，未经 sigmoid 处理。
        """
        # 使用 variable_scope 确定判别器网络的名称，以便于重用或重建。
        with tf.compat.v1.variable_scope("discriminator", reuse=reuse) as scope:
            # 如果 reuse 为 True，则重用判别器网络中的变量。
            # if reuse:
            #     scope.reuse_variables()

            # 通过卷积和 leaky ReLU 激活函数构建判别器的隐藏层。
            h0 = lrelu(conv2d(X, self.df_dim, k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(
                conv2d(h0, self.df_dim * 2, k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(
                conv2d(h1, self.df_dim * 4, k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(
                conv2d(h2, self.df_dim * 8, k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='d_h3_conv')))

            # 将卷积层的输出展平，并通过线性层减少维度，最终输出一个标量值。
            if text:
                h4 = linear(tf.reshape(h3, [2, -1]), 1, 'd_h3_lin')
                h5 = linear(tf.reshape(h3, [2, -1]), 4, 'd_h4_lin')
            else:
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                h5 = linear(tf.reshape(h3, [self.batch_size, -1]), 4, 'd_h4_lin')


            # 返回 sigmoid 激活函数处理后的输出，用于判断输入图像的真实性。
            return tf.nn.sigmoid(h4), h4, tf.nn.sigmoid(h5)

    def generator(self, z, y=None, reuse=False):
        """
        构造生成器网络。

        该函数定义了生成器的网络结构，它将隐向量z转换为输出图像。生成器通过一系列的反卷积操作，
        逐渐增加图像的尺寸，最终生成与目标尺寸相同的图像。

        参数:
          z: 输入的隐向量，形状为(batch_size, z_dim)。
          y: (可选)条件标签，形状为(batch_size, c_dim)。在无条件生成对抗网络中，可以不使用此参数。
          reuse: 是否重用变量。在同一个图中多次调用生成器时，如果reuse为True，则会重用之前定义的变量。

        返回:
          生成的图像，形状为(batch_size, output_height, output_width, c_dim)。
        """
        # 使用变量作用域"generator"来组织生成器内的变量
        with tf.compat.v1.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            # 计算输出图像在不同层的尺寸
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, stride), conv_out_size_same(s_w, stride)
            s_h4, s_w4 = conv_out_size_same(s_h2, stride), conv_out_size_same(s_w2, stride)
            s_h8, s_w8 = conv_out_size_same(s_h4, stride), conv_out_size_same(s_w4, stride)
            s_h16, s_w16 = conv_out_size_same(s_h8, stride), conv_out_size_same(s_w8, stride)

            # 将隐向量z映射到生成器的初始特征图
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            # 将初始特征图重塑为合适的形状，并应用ReLU激活函数
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            # 通过一系列的反卷积操作，逐渐增加特征图的尺寸
            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], k_h=filterH,
                                                     k_w=filterW, d_h=stride, d_w=stride, name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], k_h=filterH,
                                                k_w=filterW, d_h=stride, d_w=stride, name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], k_h=filterH,
                                                k_w=filterW, d_h=stride, d_w=stride, name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], k_h=filterH, k_w=filterW,
                                                d_h=stride, d_w=stride, name='g_h4', with_w=True)

            # 应用tanh激活函数，将输出归一化到[-1, 1]之间
            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        """
        根据给定的潜在变量z生成样本。

        参数:
        z: 一个tensor，形状为[self.z_dim]，表示潜在变量。
        y: (可选)一个tensor，提供条件信息。在这个实现中未使用。

        返回:
        一个tensor，表示生成的样本，形状为[self.output_height, self.output_width, self.c_dim]。
        """
        # 在变量作用域"generator"下操作，重用变量
        with tf.compat.v1.variable_scope("generator") as scope:
            scope.reuse_variables()

            # 计算输出尺寸的缩小比例
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, stride), conv_out_size_same(s_w, stride)
            s_h4, s_w4 = conv_out_size_same(s_h2, stride), conv_out_size_same(s_w2, stride)
            s_h8, s_w8 = conv_out_size_same(s_h4, stride), conv_out_size_same(s_w4, stride)
            s_h16, s_w16 = conv_out_size_same(s_h8, stride), conv_out_size_same(s_w8, stride)

            # 将潜在变量z映射到生成器的初始特征图
            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                            [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            # 通过一系列的反卷积操作恢复尺寸，并应用ReLU激活函数
            h1 = deconv2d(h0, [1, s_h8, s_w8, self.gf_dim * 4], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride,
                          name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [1, s_h4, s_w4, self.gf_dim * 2], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride,
                          name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [1, s_h2, s_w2, self.gf_dim * 1], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride,
                          name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [1, s_h, s_w, self.c_dim], k_h=filterH, k_w=filterW, d_h=stride, d_w=stride, name='g_h4')

            # 应用tanh激活函数，输出生成的样本
            return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        """
        获取模型存储目录的路径。

        该路径由数据集名称、批处理大小、输出高度和输出宽度组成，用于确保模型的存储目录
        能够根据训练配置唯一确定。这样可以方便地根据不同的训练参数来组织和管理模型文件。

        返回:
            模型存储目录的字符串路径，由数据集名称、批处理大小、输出高度和输出宽度拼接而成。
        """
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        # 定义模型文件名
        model_name = "VAEGAN.model"
        # 构建模型保存目录
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # 使用exist_ok=True避免抛出已存在目录的异常
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 定义模型保存路径
        model_path = os.path.join(checkpoint_dir, model_name)

        # 捕获保存过程中可能发生的任何异常
        try:
            # 保存模型到指定目录
            self.saver.save(self.sess, model_path, global_step=step)
            print(f"Model saved in path: {model_path}")
        except Exception as e:
            print(f"Error occurred while saving the model: {e}")

        # # # 如果需要删除旧的模型文件，可以取消下面代码的注释
        # if os.path.exists(model_path):
        #     os.remove(model_path)

    def load(self, checkpoint_dir):
        # 导入正则表达式模块，用于提取checkpoint名称中的步骤数
        import re
        print(" [*] Reading checkpoints...")
        # 构建模型加载目录
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        # 获取checkpoint状态
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # 如果checkpoint存在
        if ckpt and ckpt.model_checkpoint_path:
            # 提取checkpoint文件名
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # 加载模型
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # 从checkpoint文件名中提取步骤数
            counter = int(next(re.finditer(r"(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            # 返回加载成功及步骤数
            return True, counter
        else:
            # 打印未找到checkpoint的信息
            print(" [*] Failed to find a checkpoint")
            # 返回加载失败及步骤数0
            return False, 0
