# -*- coding: utf-8 -*-

########## discriminator.py ##########
#
# DCGAN Discriminator
# 
#
# created 2018/10/07 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf
import functions as f

import tensorflow as tf
import numpy as np

import os.path
import io
import math
import random
import zipfile

logger = cf.LOGGER

class Discriminator:
    '''DCGANのDiscriminatorを記述するクラス'''


    def __init__(self, global_step):
        self.global_step_tensor = global_step

        self.from_dataset = None # データセットからの入力画像
        self.from_generator = None # generatorからの入力画像
        self.output = None # 出力の確率
        self.optimizer = None
        self.train_op = None

        self.input_selector = None # generatorからの時は True

    def define_forward(self, input, vreuse = None):
        '''判定する計算を返す'''
        
        with tf.variable_scope('D_network', reuse=vreuse):

            inreshaped = tf.reshape(input,
                                    shape = (-1, cf.PIXELSIZE, cf.PIXELSIZE, 1),
                                    name = 'D_inreshaped')

            c1 = 'D_conv1'
            conv1 = f.apply_dobn(tf.layers.conv2d(inputs = inreshaped,
                                                  filters = 1,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  activation = tf.nn.leaky_relu,
                                                  name = c1),
                                 c1)
            f.print_shape(conv1)

            c2 = 'D_conv2'
            conv2 = f.apply_dobn(tf.layers.conv2d(inputs = conv1,
                                                  filters = 64,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  activation = tf.nn.leaky_relu,
                                                  name = c2),
                                 c2)
            f.print_shape(conv2)

            c3 = 'D_conv3'
            conv3 = f.apply_dobn(tf.layers.conv2d(inputs = conv2,
                                                  filters = 128,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  activation = tf.nn.leaky_relu,
                                                  name = c3),
                                 c3)
            f.print_shape(conv3)

            c4 = 'D_conv4'
            conv4 = f.apply_dobn(tf.layers.conv2d(inputs = conv3,
                                                  filters = 256,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  activation = tf.nn.leaky_relu,
                                                  name = c4),
                                 c4)
            f.print_shape(conv4)

            flatten = tf.layers.flatten(conv4,
                                        name = 'D_flatten')
            f.print_shape(flatten)

            fc = 'D_fully_connected'
            fully_connected = f.apply_dobn(tf.layers.dense(inputs = flatten,
                                                           units = 1,
                                                           activation = tf.nn.sigmoid,
                                                           name = fc),
                                           fc)
            f.print_shape(fully_connected)
            return fully_connected

    def define_graph(self):
        '''discriminatorの計算グラフを定義する'''

        with tf.variable_scope('D_network'):

            self.from_dataset = tf.placeholder(dtype=tf.float32, shape=(None, cf.PIXELSIZE, cf.PIXELSIZE), name='D_input_image')
        
            epsilon = 0.0000001
            random1 = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
                                       mean=0.0,
                                       stddev=0.015,
                                       dtype=tf.float32,
                                       name='random_1')
            random2 = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
                                       mean=0.0,
                                       stddev=0.015,
                                       dtype=tf.float32,
                                       name='random_2')
            self.p_real = tf.maximum(tf.add(self.define_forward(self.from_dataset,
                                                                vreuse=False), random1),
                                     epsilon, name='p_real')
            self.p_fake = self.define_forward(self.from_generator, vreuse=True)
            self.p_fake_for_loss = tf.maximum(tf.add(self.p_fake, random2), epsilon,
                                              name='p_fake_for_loss')

            self.loss = -tf.reduce_mean(tf.log(0.0001 + self.p_real) - tf.log(0.0001 + self.p_fake_for_loss), name='D_loss')

            D_vars = [x for x in tf.trainable_variables() if 'D_' in x.name]
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = cf.LEARNING_RATE,
                                                    beta1 = cf.BETA_1,
                                                    name = 'D_optimizer')
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step_tensor,
                                                    var_list=D_vars,
                                                    name='D_train_op')

    def set_input_from_generator(self, generator):
        with tf.variable_scope('D_network'):
            self.from_generator = generator.output
        return

    @staticmethod
    def create_minibatch():
        '''データセットからミニバッチを作成する'''
        zippath = os.path.join(cf.DATASET_PATH, cf.TRAIN_PREFIX + '.zip')
        with zipfile.ZipFile(zippath, 'r') as zf:

            nplist = zf.namelist()
            npsampled = random.sample(nplist, cf.MINIBATCHSIZE)

            minibatch = []

            for i in range(cf.MINIBATCHSIZE):
                bytes = zf.read(npsampled[i])
                buf = io.BytesIO(bytes)
                minibatch.append(np.load(buf))
        return minibatch
        