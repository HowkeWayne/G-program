# -*- coding: utf-8 -*-
#   Author: HowkeWayne
#     Date: 2019/4/22 - 9:52
"""
File Description...

"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = os.path.join(os.getcwd(), 'logs')
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 500

'''
Visualisation helper functions
Mentioned above are the sprites. If you don’t load sprites each digit is represented as a simple point (does not give 
you a lot of information). To add labels you have to create a ‘sprite map’: basically all images in what you want to 
visualise…

There are three functions which are quite important for the visualisation:

create_sprite_image: neatly aligns image sprits on a square canvas, as specified in the images section here: 
(https://www.tensorflow.org/get_started/embedding_viz)
vector_to_matrix_mnist: MNIST characters are loaded as a vector, not as an image… this function turns them into images 
invert_grayscale: matplotlib treats a 0 as black, and a 1 as white. The tensorboard embeddings visualisation looks way 
better with white backgrounds, so we invert them for the visualisation
'''


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


if __name__ == '__main__':
    path_for_mnist_sprites = os.path.join(LOG_DIR, 'mnistdigits.png')
    path_for_mnist_metadata = os.path.join(LOG_DIR, 'metadata.tsv')

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

    embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = path_for_mnist_metadata  # 'metadata.tsv'

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = path_for_mnist_sprites  # 'mnistdigits.png'
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

    to_visualise = batch_xs
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    with open(path_for_mnist_metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(batch_ys):
            f.write("%d\t%d\n" % (index, label))
