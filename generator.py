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
        inicons = 1.2


        with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

            self.latent = tf.placeholder(dtype=tf.float32,
                                         shape=(None, cf.LATENT_VECTOR_SIZE),
                                         name='G_latent_vector')
            f.print_shape(self.latent)

            pjname = 'G_projected'
            projected = f.apply_dobn(tf.layers.dense(inputs = self.latent,
                                                     units = projectedunits,
                                                     kernel_initializer = tf.initializers.random_uniform(minval=-1.0/projectedunits, maxval=1.0/projectedunits),
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
                                                             trainable = True,
                                                             activation = tf.nn.relu,
                                                             kernel_initializer = tf.initializers.random_uniform(minval=-inicons, maxval=inicons),
                                                             name = tc1),
                                  tc1)
            f.print_shape(tconv1)
            tf.summary.tensor_summary(name='G_tconv1_kernel',
                                      tensor=tf.get_variable('G_tconv1/kernel'))
            
            tc2 = 'G_tconv2'
            tconv2 = f.apply_dobn(tf.layers.conv2d_transpose(inputs = tconv1,
                                                             filters = 256,
                                                             kernel_size = (5, 5),
                                                             strides = (2, 2),
                                                             padding = 'same',
                                                             trainable = True,
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
                                                             trainable = True,
                                                             activation = tf.nn.tanh,
                                                             kernel_initializer = tf.initializers.random_uniform(minval=-1.0, maxval=1.0),
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
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=cf.G_LEARNING_RATE,
                                                    beta1 = cf.BETA_1,
                                                    name='G_optimizer')


            #self.pretrain_optimizer = tf.train.AdamOptimizer(learning_rate = 132.5,
            #                                                 name = 'G_pretrain_optimizer')
        
            pretrain_target = tf.constant(127.5,
                                          dtype = tf.float32,
                                          shape = (cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE, 1))
            self.pretrain_loss = tf.losses.mean_squared_error(labels = pretrain_target,
                                                              predictions = self.output)
        
        
            self.D.set_input_from_generator(self)
        return


    def define_graph_postD(self):

        #with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

        onesrandom = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1),
                                      mean = 1.0,
                                      stddev = 0.015,
                                      dtype = tf.float32,
                                      name='G_onesrandom')
        
        G_crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=onesrandom, logits=self.D.p_fake)

        #self.loss = tf.reduce_mean(-tf.log(0.0001 + self.D.p_fake_for_loss), name='G_loss')
        self.loss = tf.reduce_mean(G_crossentropy, name='G_loss')
        G_vars = [x for x in tf.trainable_variables() if 'G_' in x.name]
        logger.info('G_vars: ' + str(len(G_vars)))
        for v in G_vars:
            logger.info(str(v))
            
            
            #self.pretrain_step = tf.Variable(0, dtype=tf.int32,
            #                            trainable = False,
            #                            name='G_pretrain_step')

        #self.pretrain_op = self.pretrain_optimizer.minimize(self.pretrain_loss,
        #                                                    global_step = self.global_step_tensor,
        #                                                    var_list = G_vars,
        #                                                    name = 'G_pretrain_op')
        zero_tensor = tf.constant(0, dtype=self.global_step_tensor.dtype, shape=self.global_step_tensor.shape, name='zero_tensor')
        self.reset_global_step_op = tf.assign(self.global_step_tensor, zero_tensor, name='G_reset_global_step_op')
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor, var_list=G_vars, name='G_train_op')
        tf.summary.scalar(name = 'Generator loss', tensor = self.loss)
                              
        self.mean_D_score = tf.reduce_mean(self.D.p_fake)
        return

#    @staticmethod
#    def generate_latent_vector():
#        '''numpy形式でlatent vectorをランダム生成する
#        出力の値域は[0, 1]'''
#        return np.random.rand(1, cf.LATENT_VECTOR_SIZE)
        

