import math
import numpy as np

from tensorflow.python.framework import ops

from utils import *
import tensorflow as tf

# 尝试导入TensorFlow中与摘要相关的函数和类
# TensorBoard图像摘要函数
image_summary = tf.summary.image

# TensorBoard标量摘要函数
scalar_summary = tf.summary.scalar

# TensorBoard直方图摘要函数
histogram_summary = tf.summary.histogram

# 在TensorFlow 2.x中，没有merge_all函数，而是使用tf.summary.merge_all代替
merge_all = tf.compat.v1.summary.merge_all

# TensorFlow 2.x中SummaryWriter被重命名为tf.summary.create_file_writer
summary_writer = tf.summary.create_file_writer

# 检查tf.concat_v2函数是否存在，如果存在，则定义一个concat函数
if "concat_v2" in dir(tf):
    # 合并多个张量的函数
    # 参数tensors：要合并的张量列表
    # 参数axis：合并的轴
    # 返回值：合并后的张量
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)
# 如果tf.concat_v2不存在，则使用相同的方式定义concat函数
else:
    # 合并多个张量的函数
    # 参数tensors：要合并的张量列表
    # 参数axis：合并的轴
    # 返回值：合并后的张量
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
    """
    批量归一化层类。

    该类用于实现批量归一化功能，它可以被添加到神经网络的层中，以稳定训练过程并加速收敛。

    参数:
    - epsilon: 一个小的正数，用于避免分母为零的情况。
    - momentum: 动量参数，用于计算移动平均。
    - name: 该层的名称，用于标识和区分不同的批量归一化层。
    """

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        初始化批量归一化层。

        设置层的基本参数，包括epsilon、momentum和name。
        """
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        """
        执行批量归一化操作。
        该方法重写了类实例的调用操作符，使得可以通过`layer(x)`的方式来应用批量归一化。

        参数:
        - x: 输入的张量，需要进行批量归一化的数据。
        - train: 布尔值，指示当前是否处于训练模式。默认为True。

        返回:
        - 经过批量归一化处理后的张量。
        """
        return tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                  epsilon=self.epsilon,
                                                  scale=True,
                                                  name=self.name)(x)


def conv_cond_concat(x, y):
    """
    沿特征图轴连接两个张量。

    此函数用于将条件向量y与特征图x进行连接，以便将条件信息应用于特征图的每个位置。这在条件生成对抗网络（cGANs）中常见。

    参数:
    x: 特征图张量，是一个四维张量。
    y: 条件张量，也是一个四维张量。其通道维度（最后一个维度）的大小可能与x不同。

    返回值:
    连接后的张量，除了通道维度外，其形状与x相同，其中通道尺寸是x和y通道尺寸的总和。
    """
    # 获取张量x的形状
    x_shapes = x.get_shape()
    # 获取张量y的形状
    y_shapes = y.get_shape()
    # 沿通道维度（axis=3）连接x和y
    # 首先，扩展y的维度以匹配x的形状，然后进行连接
    return concat([
        x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


# stride: d_h,d_w: 1 , filter size: k_h*k_w (3*3) , #filters: output_dim: df_dim*2, *4, *8
def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    """
    实现一个2D卷积层。

    参数:
    input_ -- 输入的4D张量，格式为[batch_size, height, width, channels]。
    output_dim -- 卷积层输出的特征图的通道数。
    k_h -- 卷积核的高度。
    k_w -- 卷积核的宽度。
    d_h -- 卷积的垂直步长。
    d_w -- 卷积的水平步长。
    stddev -- 卷积核初始化的标准差。
    name -- 卷积层的名称。

    返回:
    conv -- 经过卷积和偏置添加后的4D张量。
    """
    # 使用变量作用域为卷积层命名，允许重用。
    with tf.compat.v1.variable_scope(name) as scope:
        # 初始化卷积核权重。
        # filter (i.e., w) : [filter_height, filter_width, in_channels, out_channels]
        # input_get_shape()[-1]: if the input is of size (64,88,16,1) --> 1
        # output_dim: 64
        w = tf.compat.v1.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                      initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        # 应用2D卷积操作。
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        # 初始化偏置项。
        biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # 将偏置项添加到卷积结果上，并重塑结果以保持与tf.nn.bias_add的兼容性。
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    """
    实现二维转置卷积（也称为逆卷积）操作。

    参数:
    input_ -- 输入的张量，格式为[batch_size, height, width, channels]。
    output_shape -- 输出张量的形状，不包括batch_size。
    k_h -- 卷积核的高度。
    k_w -- 卷积核的宽度。
    d_h -- 卷积核在高度方向上的步长。
    d_w -- 卷积核在宽度方向上的步长。
    stddev -- 卷积核参数的初始化标准差。
    name -- 操作的名称。
    with_w -- 是否返回卷积核参数。

    返回:
    如果with_w为True，则返回转置卷积的结果、卷积核权重和偏置项；
    如果with_w为False，则只返回转置卷积的结果。
    """
    with tf.compat.v1.variable_scope(name):
        # 初始化卷积核权重
        # filter : [height, width, output_channels, in_channels]
        w = tf.compat.v1.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                      initializer=tf.random_normal_initializer(stddev=stddev))

        # 尝试使用新版本TensorFlow的转置卷积函数
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        # 如果版本不支持，使用旧版本的函数
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # 初始化偏置项
        biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # 将偏置项添加到转置卷积结果上，并重塑结果以匹配正确的形状
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        # 根据with_w参数决定是否返回卷积核和偏置项
        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    """
    实现 leaky ReLU 激活函数。

    leaky ReLU 是 ReLU 的变体，允许一定程度的负值通过，避免“死亡ReLU”问题。

    参数:
    - x: 输入的张量。
    - leak: 负值区间的斜率，默认为0.2。
    - name: 操作的名称，用于TensorFlow图中的标识。

    返回:
    - 处理后的张量，应用了leaky ReLU函数。
    """
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    实现全连接层的线性变换。

    参数:
    - input_: 输入的张量，其最后一维应该是输入特征的数量。
    - output_size: 输出张量的特征维度。
    - scope: 变量作用域的名称，用于TensorFlow图中的标识。
    - stddev: 权重初始化的标准差。
    - bias_start: 偏置项的初始值。
    - with_w: 是否返回权重矩阵，默认为False。

    返回:
    - 如果 with_w 为True，则返回线性变换的结果、权重矩阵和偏置项；
    - 如果 with_w 为False，则只返回线性变换的结果。
    """
    shape = input_.get_shape().as_list()

    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                           tf.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
                                         initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
