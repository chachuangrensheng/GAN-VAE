"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import imageio
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf

# import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.compat.v1.trainable_variables()
    with tf.compat.v1.Session() as sess:  # 创建一个 TensorFlow session
        sess.run(tf.compat.v1.global_variables_initializer())  # 初始化变量
        # 遍历并打印每个变量的信息
        for var in model_vars:
            print(f"Variable name: {var.name}")
            print(f"Shape: {var.shape}")
            value = sess.run(var)  # 使用 session 来获取变量的值
            print(f"Value: {value}")


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    """
    从给定路径加载图像，并对其进行预处理。

    该函数首先读取指定路径下的图像，然后根据输入参数对图像进行预处理，
    包括调整图像大小、裁剪图像以及转换为灰度图像等操作。

    参数:
    image_path: 字符串，图像文件的路径。
    input_height: 整数，输入模型的图像高度。
    input_width: 整数，输入模型的图像宽度。
    resize_height: 整数，默认为64，指定图像的缩放高度。
    resize_width: 整数，默认为64，指定图像的缩放宽度。
    crop: 布尔值，默认为True，指定是否裁剪图像。
    grayscale: 布尔值，默认为False，指定是否将图像转换为灰度。

    返回:
    预处理后的图像。
    """
    # 从指定路径加载图像，根据grayscale参数决定是否以灰度模式加载
    image = imread(image_path, grayscale)
    # 对加载的图像进行预处理，包括调整大小、裁剪等，以满足模型输入要求
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)



def save_images(images, size, image_path):
    """
    保存处理后的图像。

    参数:
    images: 处理后的图像数组。
    size: 图像的尺寸。
    image_path: 图像保存的路径。

    返回:
    保存图像后的结果。
    """
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    """
    读取图像文件。

    参数:
    path: 图像文件的路径。
    grayscale: 是否以灰度模式读取图像，默认为False。

    返回:
    读取后的图像数组。
    """
    if (grayscale):
        # return scipy.misc.imread(path, flatten=True).astype(np.float)
        return imageio.imread(path, as_gray=True).astype(np.float)

    else:
        return imageio.imread(path).astype(np.float)


def merge_images(images, size):
    """
    合并多个图像为一个大图像。

    参数:
    images: 要合并的图像数组。
    size: 合并后的图像尺寸。

    返回:
    合并后的图像数组。
    """
    return inverse_transform(images)


def merge(images, size):
    """
    按照给定的尺寸合并多个图像。

    参数:
    images: 要合并的图像数组。
    size: 合并后的图像的行数和列数。

    返回:
    合并后的图像数组。

    异常:
    ValueError: 如果图像的维度不符合要求，抛出ValueError异常。
    """
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    """
    将处理后的图像保存到指定路径。

    参数:
    images: 图像数组，包含多个待合并的图像。
    size: 合并后图像的尺寸，指定合并后的行数和列数。
    path: 图像保存的路径。

    返回:
    bool值，表示图像保存是否成功。
    """
    # 合并图像数组为单个大图像
    image = np.squeeze(merge(images, size))
    # 将合并后的图像写入指定路径
    return imageio.imwrite(path, image)


# def center_crop(x, crop_h, crop_w,
#                 resize_h=64, resize_w=64):
#     if crop_w is None:
#         crop_w = crop_h
#     h, w = x.shape[:2]
#     j = int(round((h - crop_h) / 2.))
#     i = int(round((w - crop_w) / 2.))
#     return scipy.misc.imresize(
#         x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])

def center_crop(x, crop_h, crop_w=None, resize_h=64, resize_w=64):
    """
    从输入图像中中心裁剪指定大小的区域，并将裁剪后的区域缩放到指定大小。

    参数:
    x: 输入图像，可以是TensorFlow张量或任何支持形状操作的图像数据。
    crop_h: 裁剪区域的高度。
    crop_w: 裁剪区域的宽度。如果未指定，则与crop_h相同。
    resize_h: 缩放后的高度。
    resize_w: 缩放后的宽度。

    返回:
    缩放后的裁剪图像。
    """
    # 如果未指定裁剪宽度，则等于裁剪高度
    if crop_w is None:
        crop_w = crop_h
    # 获取输入图像的高度和宽度
    h, w = x.shape[:2]
    # 计算裁剪区域的起始坐标
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    # 执行中心裁剪
    # 首先进行中心裁剪
    cropped = x[j:j + crop_h, i:i + crop_w]
    # 对裁剪后的区域进行缩放
    # 然后进行缩放
    resized = tf.image.resize(cropped, [resize_h, resize_w], method='bilinear')
    return resized


# def transform(image, input_height, input_width,
#               resize_height=64, resize_width=64, crop=True):
#     if crop:
#         cropped_image = center_crop(
#             image, input_height, input_width,
#             resize_height, resize_width)
#     else:
#         cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
#     return np.array(cropped_image) / 127.5 - 1.


# 图像转换函数，根据输入参数对图像进行裁剪、缩放和归一化
def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    # 根据裁剪参数裁剪图像或直接缩放图像
    if crop:
        # 使用 TensorFlow 函数进行中心裁剪和缩放
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        # 使用 TensorFlow 函数进行缩放
        cropped_image = tf.image.resize(image, [resize_height, resize_width], method='bilinear')

    # 将图像数据归一化到 [-1, 1]
    # 将图像数据归一化到 [-1, 1]
    normalized_image = np.array(cropped_image) / 127.5 - 1

    return normalized_image

# 反转图像转换，将归一化的图像数据转换回原始范围
def inverse_transform(images):
    return (images + 1.) / 2.

# 将模型的权重和偏差转换为 JSON 格式，用于前端展示
def to_json(output_path, *layers):
    # 打开输出文件，准备写入 JSON 数据
    with open(output_path, "w") as layer_f:
        lines = ""
        # 遍历每一层权重、偏差和批量归一化数据
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            # 判断当前层是否为全连接层
            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            # 构建偏差参数的字典
            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            # 如果存在批量归一化，构建 gamma 和 beta 参数的字典
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            # 根据层类型（全连接层或卷积层）构建 JSON 字符串
            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        # 将构建好的 JSON 字符串写入文件
        layer_f.write(" ".join(lines.replace("'", "").split()))



# 定义一个函数，用于生成GIF动画
def make_gif(images, fname, duration=2, true_image=False):
    """
    生成GIF动画。

    参数:
    images: 图像列表，用于生成动画的帧。
    fname: 字符串，保存的GIF文件名。
    duration: 动画的总时长，单位为秒，默认为2秒。
    true_image: 布尔值，如果为True，则直接返回图像，否则对图像进行处理后返回。
    """
    # 导入moviepy.editor模块，用于视频剪辑
    import moviepy.editor as mpy
    # 使用moviepy库创建一个VideoClip对象
    # 根据当前时间计算帧
    def make_frame(t):
        # 根据时间计算当前帧的索引
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        # 如果true_image为True，直接返回图像
        if true_image:
            return x.astype(np.uint8)
        else:
            # 否则，对图像进行处理后返回
            return ((x + 1) / 2 * 255).astype(np.uint8)

    # 创建一个VideoClip对象，使用make_frame函数生成每一帧
    clip = mpy.VideoClip(make_frame, duration=duration)
    # 将VideoClip对象写入GIF文件
    clip.write_gif(fname, fps=len(images) / duration)

# 定义一个函数，用于可视化生成的图像
def visualize(sess, dcgan, config, option):
    """
    可视化生成的图像。

    参数:
    sess: TensorFlow会话。
    dcgan: DCGAN模型。
    config: 配置信息。
    option: 可视化选项。
    """
    # 计算图像帧的大小
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    # 根据选项进行不同的可视化操作
    if option == 0:
        # 生成随机z向量，用于生成样本
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        # 通过会话运行生成器，获取样本图像
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        # 保存样本图像
        save_images(samples, [image_frame_dim, image_frame_dim],
                    './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 1:
        # 生成0到1之间的平滑变化值
        values = np.arange(0, 1, 1. / config.batch_size)
        # 循环生成样本
        for idx in xrange(100):
            print(" [*] %d" % idx)
            # 初始化z向量
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            # 根据索引和值，填充z向量
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            # 根据数据集类型，生成样本
            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            # 保存样本图像
            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        # 生成0到1之间的平滑变化值
        values = np.arange(0, 1, 1. / config.batch_size)
        # 随机选择索引，生成样本
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            # 初始化z向量
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            # 根据数据集类型，生成样本
            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            # 尝试生成GIF动画，如果失败则保存为静态图像
            try:
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [image_frame_dim, image_frame_dim],
                            './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 3:
        # 生成0到1之间的平滑变化值
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            # 初始化z向量
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            # 生成样本
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            # 生成GIF动画
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        # 初始化图像集
        image_set = []
        # 生成0到1之间的平滑变化值
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            # 初始化z向量
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            # 将生成的样本添加到图像集中
            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            # 生成GIF动画
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        # 合并图像集中的图像，生成新的GIF动画
        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in list(range(64)) + list(range(63, -1, -1))]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
