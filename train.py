# -*- coding: utf-8 -*-

########## train.py ##########
#
#
# DCGAN
# ネットワークの学習を行うメインプログラム
#
# created 2018/10/07 Takeyuki Watadani
#
########################################

import config as cf
import functions as f
import discriminator as d
import generator as g

import random
import os.path
import math
import traceback
from time import sleep
import tensorflow as tf
import numpy as np
from PIL import Image

logger = cf.LOGGER

class Trainer:
    '''GANの学習を行うためのクラス'''

    def __init__(self, global_step, discriminator, generator):
        self.D = discriminator
        self.G = generator
        #self.trainq = queue.Queue(maxsize = cf.QUEUEMAX)
        self.training = False
        self.global_step_tensor = global_step
        self.session = None
        self.chkpdir = os.path.join(cf.RESULTDIR, cf.DESC)
        self.saver = tf.train.Saver()
        self.tf_initialize()


    def tf_initialize(self):
        '''TensorFlow関係の初期化'''
        tf.global_variables_initializer()

    def train(self):
        '''学習を行うメインルーチン
        ここからqueuingスレッドと学習を行う本スレッドに分岐する'''

        self.training = True
        try:
            #D, Gの計算グラフを定義
            self.G.define_graph()
            self.D.define_graph()
            self.G.define_graph_postD()

            self.train_loop()
        except:
            import traceback
            traceback.print_exc()
        self.training = False
        return

    def save_sample_img(self, session, 
                        total_img, save_generator_img = True):
        '''Generatorが産生するサンプルイメージを保存する'''

        if save_generator_img:
            output_batch = session.run(self.G.output)
        else:
            output_batch = self.D.create_minibatch(session)

        # 新しいImageを作る
        sampleimg = Image.new(mode='L', size=(cf.PIXELSIZE * 3, cf.PIXELSIZE * 2))

        for i in range(6):
            if cf.MINIBATCHSIZE >= i:
                img_np = output_batch[i].reshape((cf.PIXELSIZE, cf.PIXELSIZE))
                img_uint8np = np.uint8(img_np)
                img_img = Image.fromarray(img_uint8np, mode='L')

                sampleimg.paste(img_img, box=(cf.PIXELSIZE * (i % 3), cf.PIXELSIZE * (i // 3)))

        #save
        imgdir = os.path.join(cf.RESULTDIR, cf.DESC)
        filename = os.path.join(imgdir, 'sample-' + str(total_img) + '.png')
        sampleimg.save(filename)
                    

    def train_loop(self):
        '''学習を行うメインループを記述'''

        last_Dscore = 0.5

        try:
            merged = tf.summary.merge_all()

            logger.info('G optimizer variables: ' + str(self.G.optimizer.variables()))
            logger.info('D optimizer variables: ' + str(self.D.optimizer.variables()))

            with tf.Session() as self.session:

                self.summarywriter = tf.summary.FileWriter(self.chkpdir,
                                                           graph=self.session.graph)
                                                           # r1.8ではsessionなし
                                                           #,
                                                           #session=self.session)

                self.session.run([tf.global_variables_initializer(),
                                  tf.local_variables_initializer()])

                finished = False
                while not finished:
                    
                    gstep = self.session.run(self.global_step_tensor) // 2 

                    # Discriminatorの学習

                    _, last_Dscore, dloss = self.session.run([self.D.train_op,
                                                              self.G.mean_D_score, self.D.loss])

                    # Generatorの学習
                    _, last_Dscore, gloss = self.session.run([self.G.train_op,
                                                              self.G.mean_D_score,
                                                              self.G.loss])


                    if last_Dscore < 1e-2 and cf.USE_INTENSIVE_TRAINING:
                        counter = 0
                        while last_Dscore < 0.45 and counter < 50000:
                            _, last_Dscore, gloss, dloss = self.session.run([self.G.train_op,
                                                                             self.G.mean_D_score,
                                                                             self.G.loss,
                                                                             self.D.loss])

                            counter += 1
                            if counter % 10 == 0:
                                print('counter: ', counter, 'gloss:', gloss,
                                      'dloss:', dloss, 'last_Dscore:', last_Dscore)
                    #elif last_Dscore > 0.999 and cf.USE_INTENSIVE_TRAINING:
                    #    counter = 0
                    #    while last_Dscore > 0.70 and counter < 500:
                    #        dminibatch = self.obtain_minibatch
                    #        fd = { self.D.from_dataset: dminibatch}#,
                    #               #self.G.latent: gminibatch }
                    #        _, last_Dscore, dloss = self.session.run([self.D.train_op,
                    #                                                  self.G.mean_D_score,
                    #                                                  self.D.loss],
                    #                                                 feed_dict=fd)
                    #        counter += 1
                    #        if counter % 10 == 0:
                    #            print('counter: ', counter, 'gloss:', gloss, 'dloss:', dloss,
                    #                  'last_Dscore:', last_Dscore)

                    total_img = gstep * cf.MINIBATCHSIZE
                    current_epoch = total_img / cf.TRAIN_DATA_NUMBER

                    # 一定回数ごとに現状を表示
                    if gstep % 10 == 0:
                        logger.info('Epoch: %.3f, total_img: %s, Dscore: %s, dloss: %s, gloss: %s', current_epoch, total_img, last_Dscore, dloss, gloss)
                        try:
                            self.summarywriter.flush()
                        except:
                            traceback.print_exc()
                    # テスト: 一定回数毎にGのカーネル変数をチェック
                    if gstep % 25 == 0:
                        k4 = self.session.run(self.G.kernel4)#, feed_dict=fd)
                        logger.info('G_convkernel 4: mean=' + str(np.mean(k4)) + ', std=' + str(np.std(k4)))
                    # テスト: 一定回数毎にGのgradientをチェック
                    if gstep % 500 == 12:
                        gradients = self.session.run(self.G.gradients)#, feed_dict=fd)
                        for g in gradients:
                            mean = np.mean(g[0])
                            std = np.std(g[0])
                            logger.info('gradient mean: ' + str(mean) + ', gradient std: ' + str(std))

                    # 一定回数ごとにチェックポイントをセーブ
                    if gstep % 500 == 0:
                        try:
                            self.saver.save(self.session,
                                            os.path.join(self.chkpdir,'model.ckpt'),
                                            gstep)
                            logger.info('checkpointをセーブしました。')
                            summary = self.session.run(merged)#, feed_dict=fd)
                            self.summarywriter.add_summary(summary, gstep)
                        except:
                            traceback.print_exc()

                    # 一定回数ごとにサンプル画像を保存
                    if gstep % cf.SAVE_SAMPLE_MINIBATCH == 0:
                        self.save_sample_img(self.session,
                                             #fd,
                                             total_img)
                        logger.info('サンプルを保存しました。')
                    # 一定回数ごとに正解画像を保存
                    if gstep % 1000 ==0:
                        self.save_sample_img(self.session,
                                             #fd,
                                             'D_'+str(total_img), False)
                    # 最大epochに到達したら終了フラグを立てる
                    if current_epoch >= cf.MAXEPOCH:
                        logger.info('max_epochに到達したため終了します。')
                        finished = True
        except:
            traceback.print_exc()
            self.training = False
        finally:
            logger.info('train_loopを終了します。')


logger.info('DCGAN 学習プログラムを開始します。')

# 乱数を初期化
logger.info('乱数を初期化します')
random.seed()

# ネットワークのインスタンスを作成
global_step_tensor = tf.train.create_global_step()

D = d.Discriminator(global_step_tensor)
G = g.Generator(D, global_step_tensor)

T = Trainer(global_step_tensor, D, G)

logger.info('nn.trainを開始します。')
T.train()

logger.info('学習が終了しました。')
chkpdir = os.path.join(cf.RESULTDIR, cf.DESC)
chkp = tf.train.latest_checkpoint(chkpdir)
if chkpdir is not None and chkp is not None:
    logger.info('最新チェックポイント: ' + tf.train.latest_checkpoint(chkpdir))
logger.info('プログラムを終了します。')

