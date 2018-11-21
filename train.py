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
import discriminator as d
import generator as g

import random
import queue
import os.path
import math
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import numpy as np
from PIL import Image

logger = cf.LOGGER

class Trainer:
    '''GANの学習を行うためのクラス'''

    def __init__(self, global_step, discriminator, generator):
        self.D = discriminator
        self.G = generator
        self.trainq = queue.Queue(maxsize = cf.QUEUEMAX)
        self.training = False
        self.global_step_tensor = global_step
        self.session = None

        self.tf_initialize()


    def tf_initialize(self):
        '''TensorFlow関係の初期化'''
        tf.global_variables_initializer()

    def train(self, nqueuethreads=2):
        '''学習を行うメインルーチン
        ここからqueuingスレッドと学習を行う本スレッドに分岐する'''

        self.training = True
        try:
            executor = ThreadPoolExecutor()

            qfutures = []
            for i in range(nqueuethreads):
                qfutures.append(executor.submit(self.queuing_loop, i))

                #D, Gの計算グラフを定義
                #self.D.set_input_from_generator(self.G)
            self.G.define_graph()
            self.D.define_graph()
            self.G.define_graph_postD()


            self.train_loop()
        except:
            import traceback
            traceback.print_exc()
        self.training = False
        return

    def obtain_minibatch(self):
        '''学習用のキューからミニバッチデータを取り出す'''
        while self.trainq.empty():
            sleep(cf.SLEEP_INTERVAL)
        return self.trainq.get()

    def queuing_loop(self, qloopid):
        '''ミニバッチを作成し、キューに格納する処理の本体'''
    
        try:
            logger.info('queuing_loopを開始します。id=' + str(qloopid))
            if self.trainq is None:
                self.trainq = queue.Queue(maxsize=cf.QUEUEMAX)

            while self.training:
                if self.trainq.qsize() < cf.QUEUEBUFSIZE:
                    for _ in range(cf.QUEUEBATCHSIZE):
                        minibatch = self.D.create_minibatch()
                        self.trainq.put(minibatch)
                else:
                    sleep(cf.SLEEP_INTERVAL)
        except:
            import traceback
            traceback.print_exc()
        logger.info('queuing_loopを終了します。qloopid = ' + str(qloopid))

    def save_sample_img(self, session, feed_dict, total_img, save_generator_img = True):
        '''Generatorが産生するサンプルイメージを保存する'''

        if save_generator_img:
            output_batch = session.run(self.G.output, feed_dict=feed_dict)
        else:
            output_batch = feed_dict[self.D.from_dataset]


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
        chkpdir = os.path.join(cf.RESULTDIR, cf.DESC)
        last_Dscore = 0.5
        #intensive_trained = False

        #dummy_Dinput = np.zeros(shape=[cf.MINIBATCHSIZE, cf.PIXELSIZE, cf.PIXELSIZE],
        #                        dtype=np.float)

        try:
            with tf.train.MonitoredTrainingSession(checkpoint_dir = chkpdir) as self.session:

                prelosstarget = math.sqrt(1000 * cf.PIXELSIZE * cf.PIXELSIZE)
                logger.info('Generatorのpretrainを開始します 目標値 = ' + str(prelosstarget))

                gpretrainloss = 999999999999

                while not (self.session.should_stop() or gpretrainloss < prelosstarget):
                    gminibatch = []
                    for _ in range(cf.MINIBATCHSIZE):
                        gminibatch.append(self.G.generate_latent_vector())

                    gminibatch = np.concatenate(gminibatch, axis=0)
                    fd = { self.G.latent: gminibatch }

                    _, pretrain_step, gpretrainloss = self.session.run([self.G.pretrain_op, self.G.global_step_tensor, self.G.pretrain_loss], feed_dict=fd)
                    if pretrain_step % 20 == 0:
                        logger.info('Step: ' + str(pretrain_step) + ', pretrain loss: ' + str(gpretrainloss))


                logger.info('Generatorのpretrainが終了しました。これより敵対的学習に入ります。')

                logger.info('Global Stepをリセットします。')
                self.session.run(self.G.reset_global_step_op)

                finished = False
                while not (self.session.should_stop() or finished):
                    gstep = self.session.run(self.global_step_tensor) // 2

                    # feed_dictの準備
                    dminibatch = self.obtain_minibatch()

                    gminibatch = []
                    for _ in range(cf.MINIBATCHSIZE):
                        gminibatch.append(self.G.generate_latent_vector())

                    gminibatch = np.concatenate(gminibatch, axis=0)

                    fd = {
                        self.D.from_dataset: dminibatch,
                        self.G.latent: gminibatch,
                    }

                    # Discriminatorの学習

                    _, last_Dscore, dloss = self.session.run([self.D.train_op,
                                                              self.G.mean_D_score, self.D.loss],
                                                             feed_dict=fd)
                
                    # Generatorの学習
                    _, last_Dscore, gloss = self.session.run([self.G.train_op,
                                                              self.G.mean_D_score,
                                                              self.G.loss],
                                                             feed_dict=fd)

                    if last_Dscore < 1e-3 and cf.USE_INTENSIVE_TRAINING:
                        counter = 0
                        while last_Dscore < 0.30 and counter < 500:
                            gminibatch = []
                            for _ in range(cf.MINIBATCHSIZE):
                               gminibatch.append(self.G.generate_latent_vector())
                       
                            gminibatch = np.concatenate(gminibatch, axis=0)
                            fd = { self.D.from_dataset: dminibatch, self.G.latent: gminibatch }
                            _, last_Dscore, gloss, dloss = self.session.run([self.G.train_op,
                                                                             self.G.mean_D_score,
                                                                             self.G.loss,
                                                                             self.D.loss],
                                                                            feed_dict=fd)
                            counter += 1
                            if counter % 10 == 0:
                                print('counter: ', counter, 'gloss:', gloss,
                                      'dloss:', dloss, 'last_Dscore:', last_Dscore)
                    elif last_Dscore > 0.999 and cf.USE_INTENSIVE_TRAINING:
                        counter = 0
                        while last_Dscore > 0.70 and counter < 500:
                            dminibatch = self.obtain_minibatch
                            fd = { self.D.from_dataset: dminibatch,
                                   self.G.latent: gminibatch }
                            _, last_Dscore, dloss = self.session.run([self.D.train_op,
                                                                      self.G.mean_D_score,
                                                                      self.D.loss],
                                                                     feed_dict=fd)
                            counter += 1
                            if counter % 10 == 0:
                                print('counter: ', counter, 'gloss:', gloss, 'dloss:', dloss,
                                      'last_Dscore:', last_Dscore)

                    total_img = gstep * cf.MINIBATCHSIZE
                    current_epoch = total_img / cf.TRAIN_DATA_NUMBER

                    # 一定回数ごとに現状を表示
                    if gstep % 10 == 0:
                        logger.info('Epoch: %.3f, total_img: %s, Dscore: %s, dloss: %s, gloss: %s, qsize: %s', current_epoch, total_img, last_Dscore, dloss, gloss, self.trainq.qsize())
                    # 一定回数ごとにサンプル画像を保存
                    if gstep % cf.SAVE_SAMPLE_MINIBATCH == 0:
                        self.save_sample_img(self.session, fd, total_img)
                        logger.info('サンプルを保存しました。')
                    # 一定回数ごとに正解画像を保存
                    if gstep % 1000 ==0:
                        self.save_sample_img(self.session, fd, 'D_'+str(total_img), False)
                    # 最大epochに到達したら終了フラグを立てる
                    if current_epoch >= cf.MAXEPOCH:
                        logger.info('max_epochに到達したため終了します。')
                        finished = True
        except:
            import traceback
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
T.train(nqueuethreads=3)

logger.info('学習が終了しました。')
chkpdir = os.path.join(cf.RESULTDIR, cf.DESC)
chkp = tf.train.latest_checkpoint(chkpdir)
if chkpdir is not None and chkp is not None:
    logger.info('最新チェックポイント: ' + tf.train.latest_checkpoint(chkpdir))
logger.info('プログラムを終了します。')

