# -*- coding: utf-8 -*-
#   Author: HowkeWayne
#   Date: 2019/4/18 - 9:11
"""
File Description...
lenet-5 网络测试实验
LeNet5 implements on tensorflow
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TF只显示 error 信息
LOG_DIR = os.path.join(os.getcwd(), 'logs')
NAME_TO_VISUALISE_VARIABLE = "mnistEmbedding"
TO_EMBED_COUNT = 500


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits


class LeNet5:
    def __init__(self):
        print('当前tensortflow版本:{0}'.format(tf.__version__))
        print('当前keras版本:{0}'.format(tf.keras.__version__))
        # 注意path是绝对路径
        self.mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
        # Up method is deprecated,using below way(tf.keras)
        # self.mnist = tf.keras.datasets.mnist
        # (self.mnist.train_image, self.mnist.train_label), (self.mnist.test_image, self.mnist.test_label) \
        #     = self.mnist.load_data(path=os.getcwd() + r'\Data\mnist.npz')
        self.path_for_mnist_sprites = os.path.join(LOG_DIR, 'mnistdigits.png')  # 映射图片地址
        self.path_for_mnist_metadata = os.path.join(LOG_DIR, 'metadata.tsv')  # label 和 index 对应表地址
        self.summary_writer = tf.summary.FileWriter(LOG_DIR)  # 事件记录器
        self.embedding_var = tf.Variable(tf.ones([1024, 64]), name=NAME_TO_VISUALISE_VARIABLE)  # 投影变量
        self.config = projector.ProjectorConfig()  # 配置投影
        self.embedding = self.config.embeddings.add()
        self.embedding.tensor_name = self.embedding_var.name
        # Specify where you find the metadata
        self.embedding.metadata_path = self.path_for_mnist_metadata  # 'metadata.tsv'
        # Specify where you find the sprite (we will create this later)
        self.embedding.sprite.image_path = self.path_for_mnist_sprites  # 'mnistdigits.png'
        self.embedding.sprite.single_image_dim.extend([28, 28])
        # Say that you want to visualise the embeddings
        projector.visualize_embeddings(self.summary_writer, self.config)

    @staticmethod
    def softmax(x):
        """
            softmax function implements with numpy
            parameters:
                x: a numpy array
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    # LeNet-5 model
    @staticmethod
    def inference(input_tensor):
        with tf.variable_scope("layer1-conv1"):
            conv1_weight = tf.get_variable(name="conv1_variable", shape=[5, 5, 1, 6],
                                           initializer=tf.truncated_normal_initializer())
            conv1_bias = tf.get_variable(name="conv1_bias", shape=[6], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weight, strides=[1, 1, 1, 1], padding="VALID")
            relu1 = tf.nn.relu(tf.add(conv1, conv1_bias))
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        with tf.variable_scope("layer2-conv2"):
            conv2_weight = tf.get_variable(name="conv2_variable", shape=[5, 5, 6, 16],
                                           initializer=tf.truncated_normal_initializer())
            conv2_bias = tf.get_variable(name="conv2_bias", shape=[16], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weight, strides=[1, 1, 1, 1], padding="VALID")
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        with tf.variable_scope("layer3-fc1"):
            conv_layer_flatten = tf.layers.flatten(inputs=pool2)  # [batch_size, 256]
            fc1_variable = tf.get_variable(name='fc1_variable', shape=[256, 128],
                                           initializer=tf.random_normal_initializer()) * 0.01
            fc1_bias = tf.get_variable(name='fc1_bias', shape=[1, 128], initializer=tf.constant_initializer(value=0))
            fc1 = tf.nn.relu(tf.add(tf.matmul(conv_layer_flatten, fc1_variable), fc1_bias))  # [batch_size, 120]
        with tf.variable_scope("layer4-fc2"):
            fc2_variable = tf.get_variable(name="fc2_variable", shape=[128, 64],
                                           initializer=tf.random_normal_initializer()) * 0.01  # [batch_size, 84]
            fc2_bias = tf.get_variable(name="fc2_bias", shape=[1, 64], initializer=tf.constant_initializer(value=0))
            fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, fc2_variable), fc2_bias))  # [batch_size, 64]
        with tf.variable_scope("layer5-output"):
            output_variable = tf.get_variable(name="output_variable", shape=[64, 10],
                                              initializer=tf.random_normal_initializer()) * 0.01
            output_bias = tf.get_variable(name="output_bias", shape=[1, 10],
                                          initializer=tf.constant_initializer(value=0))
            output = tf.add(tf.matmul(fc2, output_variable), output_bias)  # [batch_size, 10]
        return output, fc2

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    # training model
    def train(self, iter_num=500, batch_size=400, learning_rate=0.1, learning_rate_decay=0.85):
        costs = []
        x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="x")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="y")
        output, fc2 = self.inference(x)
        assignment = self.embedding_var.assign(fc2)
        # (Softmax->交叉熵)->均值误差
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y, name="loss"))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        saver = tf.train.Saver()  # 添加参数存储器
        file_writer = tf.summary.FileWriter(LOG_DIR)
        with tf.Session() as sess:
            file_writer.add_graph(sess.graph)
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(iter_num):
                batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape([-1, 28, 28, 1])
                loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y: batch_ys})
                costs.append(loss)
                if i % 100 == 0:
                    learning_rate = learning_rate * learning_rate_decay ** (i / 100)
                    print("loss after %d iteration is : " % i + str(loss))
                    batch_xs = self.mnist.validation.images[:1024]
                    # batch_ys = self.mnist.validation.labels[:1024]
                    batch_xs = batch_xs.reshape([-1, 28, 28, 1])
                    sess.run(assignment, feed_dict={x: batch_xs, y: self.mnist.validation.labels[:1024]})
                    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)
            # saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))

        to_visualise = self.mnist.validation.images[:1024]
        labels = [np.argmax(label) for label in self.mnist.validation.labels[:1024]]
        to_visualise = vector_to_matrix_mnist(to_visualise)
        to_visualise = invert_grayscale(to_visualise)
        sprite_image = create_sprite_image(to_visualise)
        plt.imsave(self.path_for_mnist_sprites, sprite_image, cmap='gray')
        # plt.imshow(sprite_image, cmap='gray')
        with open(self.path_for_mnist_metadata, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(labels):
                f.write("%d\t%d\n" % (index, label))
        plt.figure()
        plt.title("loss")
        plt.xlabel("iteration num")
        plt.ylabel("loss")
        plt.plot(np.arange(0, iter_num), costs)
        plt.show()

    def evaluate(self, images, y_true):
        try:
            tf.reset_default_graph()
        except AssertionError:
            print('"tf.reset_default_graph()" function is called within a nested graph.')
            return
        x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        output = self.inference(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./mnistModel/model.ckpt")
            output = sess.run(output, feed_dict={x: images})
            y_pred = np.argmax(self.softmax(output), axis=1).reshape(-1, 1)
            accuracy = np.mean(y_pred == y_true)
            print("accuracy is " + str(accuracy))

    def predict(self, image):
        tf.reset_default_graph()
        image = image.reshape([1, 28, 28, 1])
        x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        predict = self.inference(x)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "./mnistModel/model.ckpt")
            predict = sess.run(predict, feed_dict={x: image})
            y_pred = np.argmax(self.softmax(predict), axis=1).reshape(-1, 1)
            print("预测结果为：" + str(y_pred))


if __name__ == "__main__":
    model = LeNet5()
    # train model
    model.train(iter_num=1000, batch_size=512, learning_rate=0.1)

    # # evaluate model on trainSet
    # images_train = model.mnist.train.images.reshape([-1, 28, 28, 1])
    # y_true_train = np.argmax(model.mnist.train.labels, axis=1).reshape(-1, 1)
    # model.evaluate(images_train, y_true_train)  # accuracy is 0.9939818181818182
    #
    # # evaluate model on testSet
    # images_test = model.mnist.test.images.reshape([-1, 28, 28, 1])
    # y_true_test = np.argmax(model.mnist.test.labels, axis=1).reshape(-1, 1)
    # model.evaluate(images_test, y_true_test)  # accuracy is 0.9897
    #
    # # evaluate model on validate
    # images_validation = model.mnist.validation.images.reshape([-1, 28, 28, 1])
    # y_true_validation = np.argmax(model.mnist.validation.labels, axis=1).reshape(-1, 1)
    # model.evaluate(images_validation, y_true_validation)  # accuracy is 0.9648
