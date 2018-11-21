#!/usr/bin/env python
# -*- coding: utf-8 -*-

########## create_dataset.py ##########
#
# DCGANのデータセット作成用ユーティリティ
# Ver. 1 ZIP形式でピクセルデータのnpyを格納する
#
# created 2018/9/21 Takeyuki Watadani @ UT Radiology
#
########################################

# import section
import config as cf

import random
import glob
import os.path
import zipfile
import tempfile

from PIL import Image
import numpy as np

##########

logger = cf.LOGGER

########## ここからメインプログラム ##########

# 乱数を初期化
random.seed()

# 教師用データセットを読み込む
logger.info('学習用データセットを作成します。')

logger.info('教師用画像サーチパス: ' + cf.TRAIN_DATA_PATH)
# 画像ファイルのリストアップ
globstr = os.path.join(cf.TRAIN_DATA_PATH, '**/*' + cf.TRAIN_DATA_EXT)
img_list = glob.glob(globstr, recursive=True)

logger.info(str(len(img_list)) + '個の画像を確認しました。画像順をシャッフルします。')
random.shuffle(img_list)

sample_number = random.randint(0, len(img_list)-1)

actual_nimg = min(len(img_list), cf.TRAIN_DATA_NUMBER) # 実際に作成するデータセットの画像数

# データセットディレクトリのチェックと作成
if not os.path.exists(cf.DATASET_PATH):
    os.makedirs(cf.DATASET_PATH)

# データセットを書き込んだ回数
record_count = 0

# zipファイルのパス名
zippath = os.path.join(cf.DATASET_PATH, cf.TRAIN_PREFIX + '.zip')

# zipファイルがすでに存在する場合はリネームする
if os.path.exists(zippath):
    zippath_new = zippath + '.old'
    os.rename(zippath, zippath_new)

while record_count < actual_nimg:

    with tempfile.TemporaryDirectory() as tmpdir:

        logger.info('テンポラリディレクトリとして' + tmpdir + 'を使用します。')

        with zipfile.ZipFile(zippath, 'w') as zf:

            for i in range(actual_nimg):
                img = Image.open(img_list[i])
                grayimg = img.convert(mode='L')
                resized = grayimg.resize((cf.PIXELSIZE, cf.PIXELSIZE), resample=Image.BICUBIC)
                uint8np = np.asarray(resized, dtype=np.uint8)
                imgnp = np.float32(uint8np)

                npyname = cf.TRAIN_PREFIX + str(i) + '.npy'
                tmpfilename = os.path.join(tmpdir, npyname)
                np.save(tmpfilename, imgnp)
                zf.write(tmpfilename, arcname=npyname)
            
                record_count += 1
                if (record_count % 100 == 0) or (record_count == actual_nimg):
                    logger.info(str(record_count) + '件まで実行しました。')
                elif record_count == sample_number:
                    sampleimg = Image.new(mode='L', size=(cf.PIXELSIZE ,
                                                          cf.PIXELSIZE))

                    print(uint8np)
                    sampleimg.paste(resized, box=(0, 0))

                    #セーブ
                    imgdir = cf.DATASET_PATH
                    filename = 'sample-' + str(sample_number) + '.png'
                    fullname = os.path.join(imgdir, filename)
                    sampleimg.save(fullname)
                    logger.info('サンプル画像を保存しました')

        logger.info(str(record_count) + '件の画像データ書き込みが終了しました。')


logger.info('データセット作成を終了します。')
