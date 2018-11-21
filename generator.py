# -*- coding: utf-8 -*-

########## generator.py ##########
#
# DCGAN Generator
# 
#
# created 2018/10/07 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf
import functions as f

import numpy as np
import tensorflow as tf

import math

logger = cf.LOGGER

class Generator:
    '''DCGANのGeneratorを記述するクラス'''

    def __init__(self, discriminator, global_step):
        self.D = discriminator # 対となるDiscriminator


        self.global_step_tensor = global_step

        self.latent = None # 入力のlatent vector
        self.output = None # 出力のピクセルデータ
        self.loss = None # 損失関数
        self.optimizer = None # オプティマイザ
        self.train_op = None # 学習オペレーション

        np.random.seed()

    def define_graph(self):
        '''generatorのネットワークを記述する'''

        projectedunits = 4 * 4 * 1024 #論文通りの数
        inicons = 10000.0 / math.sqrt(projectedunits)

        with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

            self.latent = tf.placeholder(dtype=tf.float32,
                                         shape=(None, cf.LATENT_VECTOR_SIZE),
                                         name='G_latent_vector')
            f.print_shape(self.latent)

            pjname = 'G_projected'
            projected = f.apply_dobn(tf.layers.dense(inputs = self.latent,
                                                     units = projectedunits,
                                                     kernel_initializer = tf.initializers.random_uniform(minval=-inicons, maxval=inicons),
                                                     name=pjname),
                                     pjname)
            f.print_shape(projected)

            preshaped = tf.reshape(projected,
                                   shape=(-1, 4, 4, 1024),
                                   name='G_reshaped')
            f.print_shape(preshaped)

            tc1 = 'G_tconv1'
            tconv1 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = preshaped,
                                                             filters = 512,
                                                             kernel_size = (4, 4),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.initializers.random_uniform(minval=-inicons, maxval=inicons),
                                                             name = tc1),
                                  tc1)
            f.print_shape(tconv1)

            tc2 = 'G_tconv2'
            tconv2 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv1,
                                                             filters = 256,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.initializers.random_uniform(minval=-inicons, maxval=inicons),
                                                             name = tc2),
                                  tc2)
            f.print_shape(tconv2)
            
            tc3 = 'G_tconv3'
            tconv3 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv2,
                                                             filters = 128,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.initializers.random_uniform(minval=-inicons, maxval=inicons),
                                                             name = tc3),
                                  tc3)
            f.print_shape(tconv3)

            tc4 = 'G_tconv4'
            tconv4 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv3,
                                                             filters = 1,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             activation = tf.nn.tanh,
                                                             kernel_initializer = tf.initializers.random_uniform(minval=-inicons, maxval=inicons),
                                                             name = tc4),
                                  tc4)
            f.print_shape(tconv4)

            #outreshape = tf.reshape(tconv4, shape=(-1, cf.PIXELSIZE, cf.PIXELSIZE))
            mulcons = tf.constant(2.0 / 255.0,
                                  dtype = tf.float32,
                                  shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            addcons = tf.constant(1.0,
                                  dtype = tf.float32,
                                  shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))

            #0-255の値域にする
            self.output = tf.multiply(mulcons, tf.add(addcons, tconv4),
                                      name = 'G_output')
            f.print_shape(self.output)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=cf.LEARNING_RATE,
                                                    beta1 = cf.BETA_1,
                                                    name='G_optimizer')


            self.pretrain_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001,
                                                             name = 'G_pretrain_optimizer')
            
            pretrain_target = tf.constant(127.5,
                                          dtype = tf.float32,
                                          shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            self.pretrain_loss = tf.losses.mean_squared_error(labels = pretrain_target,
                                                              predictions = self.output)


            self.D.set_input_from_generator(self)
        return


    def define_graph_postD(self):

        with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

            self.loss = tf.reduce_mean(-tf.log(0.0001 + self.D.p_fake_for_loss), name='G_loss')
            G_vars = [x for x in tf.trainable_variables() if 'G_' in x.name]
            logger.info('G_vars: ' + str(len(G_vars)))

            self.pretrain_step = tf.Variable(0, dtype=tf.int32,
                                        trainable = False,
                                        name='G_pretrain_step')

            self.pretrain_op = self.pretrain_optimizer.minimize(self.pretrain_loss,
                                                                global_step = self.pretrain_step,
                                                                var_list = G_vars,
                                                                name = 'G_pretrain_op')
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor, var_list=G_vars, name='G_train_op')
            self.mean_D_score = tf.reduce_mean(self.D.p_fake)
        return

    @staticmethod
    def generate_latent_vector():
        '''numpy形式でlatent vectorをランダム生成する
        出力の値域は[0, 1]'''
        return np.random.rand(1, cf.LATENT_VECTOR_SIZE)
        

