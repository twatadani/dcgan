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

    def define_forward(self, input, vreuse = None):
        '''判定する計算を返す'''
        
        with tf.variable_scope('D_network', reuse=vreuse):

            norm_factor = tf.constant(255.0, dtype=tf.float32,
                                      shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            inreshaped = tf.divide(tf.reshape(input,
                                              shape = (-1, cf.PIXELSIZE, cf.PIXELSIZE, 1),
                                              name = 'D_inreshaped'), norm_factor)

            c1 = 'D_conv1'
            conv1 = f.apply_dobn(tf.layers.conv2d(inputs = inreshaped,
                                                  filters = 1,
                                                  kernel_size = (4, 4),
                                                  strides = (2, 2),
                                                  padding = 'same',
                                                  kernel_initializer = tf.keras.initializers.he_uniform(),
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
                                                  kernel_initializer = tf.keras.initializers.he_uniform(),
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
                                                  kernel_initializer = tf.keras.initializers.he_uniform(),
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
                                                  kernel_initializer = tf.keras.initializers.he_uniform(),
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
                                                           kernel_initializer = tf.keras.initializers.he_uniform(),
                                                           activation = tf.nn.sigmoid,
                                                           name = fc),
                                           fc)
            f.print_shape(fully_connected)
            return fully_connected

    def define_graph(self):
        '''discriminatorの計算グラフを定義する'''

        with tf.variable_scope('D_network'):

            self.from_dataset = f.obtain_minibatch()
            print(str(self.from_dataset))
        
            zeros = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
                                     mean=0.015,
                                     stddev=0.015,
                                     dtype=tf.float32,
                                     name='D_zeros')

            onesrandom = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
                                          mean = 0.985,
                                          stddev = 0.015,
                                          dtype = tf.float32,
                                          name='onesrandom')

            self.p_fake = self.define_forward(self.from_generator, vreuse=tf.AUTO_REUSE)
            self.p_real = self.define_forward(self.from_dataset, vreuse=tf.AUTO_REUSE)


            G_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=zeros, logits=self.p_fake)
            D_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=onesrandom, logits=self.p_real)


            self.loss = tf.reduce_mean(tf.add(D_crossentropy, G_crossentropy), name='D_loss')

            tf.summary.scalar(name = 'Discriminator loss', tensor = self.loss)
            D_vars = [x for x in tf.trainable_variables() if 'D_' in x.name]
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = cf.D_LEARNING_RATE,
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
    def create_minibatch(session):
        '''データセットからミニバッチを作成する'''
        minibatch_tf = f.obtain_minibatch()
        minibatch_np = session.run(minibatch_tf)
        return minibatch_np
