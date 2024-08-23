import os
# import scipy.misc
import numpy as np

from model import VAEGAN
from utils import pp, to_json, show_all_variables

import tensorflow as tf

flags = tf.compat.v1.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "default learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 10000, "The size of train samples [np.inf]")
flags.DEFINE_integer("batch_size", 10, "The size of batch samples [64]")
flags.DEFINE_integer("input_height", 125, "The size of sample to use 125.")
flags.DEFINE_integer("input_width", 187, "The size of sample to use. If None, same value as input_height [187]")
flags.DEFINE_integer("output_height", 125, "The size of the output samples to produce [125]")
flags.DEFINE_integer("output_width", 187, "The size of the output samples to produce. If None, same value as output_height [187]")
flags.DEFINE_string("dataset", "data", "The name of dataset.")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of inputs [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the sample samples [samples]")
# flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def main(_):
    """
    程序的主入口函数。

    参数:
    _ : 标准库函数Unused变量，通常在使用Google Flags时作为函数参数，但本程序中未使用。
    """
    # 打印所有定义的标志（flags）
    pp.pprint(flags.FLAGS.__flags)

    # # 确保输入宽度和高度一致，如果未指定输出宽度，则与高度相同
    # if FLAGS.input_width is None:
    #     FLAGS.input_width = FLAGS.input_height
    # if FLAGS.output_width is None:
    #     FLAGS.output_width = FLAGS.output_height

    # 创建检查点和样本目录，如果它们不存在
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # 配置TensorFlow会话，允许GPU内存增长
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True


    # 使用配置的会话创建VAEGAN模型
    with tf.compat.v1.Session(config=run_config) as sess:
        with tf.compat.v1.variable_scope("vaegan", reuse=tf.compat.v1.AUTO_REUSE):
            vaegan = VAEGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir,
                flags=FLAGS)

            # 显示所有变量
            # show_all_variables()

            # 根据标志决定是训练还是测试模型
            if FLAGS.train:
                vaegan.train(FLAGS)
            else:
                # 如果模型未加载成功，则抛出异常
                if not vaegan.load(FLAGS.checkpoint_dir)[0]:
                    raise Exception("[!] Train a model first, then run test mode")
                else:
                    vaegan.test(FLAGS)


if __name__ == '__main__':
    tf.compat.v1.app.run()

# nvidia-smi -l 0.2
