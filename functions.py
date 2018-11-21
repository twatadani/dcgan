#! /usr/bin/env python
# -*- coding: utf-8 -*-

##########functions.py ##########
#
#
# DCGAN用のユーティリティ－関数群
#
# created 2018/10/08 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf

import tensorflow as tf

logger = cf.LOGGER

def print_shape(layer):
    '''loggerからtensorflowのlayer shapeを出力する'''
    logger.info(layer.name + ': ' + str(layer.shape))
    return

def apply_dropout(layer):
    '''layerに対してdropoutを設定する。返り値はdropout layer'''
    return tf.layers.dropout(inputs = layer,
                             rate = cf.DROPOUT_RATE,
                             name = 'dropout_' + layer.name)

def apply_batchnorm(layer):
    '''layerに対してbatch normalizationを設定する。返り値はbn layer'''
    return tf.layers.batch_normalization(inputs = layer,
                                         name = 'batchnorm_' + layer.name)

def apply_dobn(layer, basename):
    '''layerに対してdropout, batch_normalizationを設定する。
    返り値は出力となるlayer'''

    do = tf.layers.dropout(inputs = layer,
                           rate = cf.DROPOUT_RATE,
                           name = 'dropout_' + str(basename))
    bn = tf.layers.batch_normalization(inputs = do,
                                       name = 'batchnorm_' + str(basename))
    return bn

                             


