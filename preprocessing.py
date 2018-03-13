#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# name: demo
# author: 王曦（ Wang Xi ）
# author_email: buaawangxi@buaa.edu.cn
# date: 2018/3/8
# time: 14:43

import os
import shutil
import keras

train_filenames = os.listdir('row_data/train')
train_cat = filter(lambda x: x[:3]=='cat', train_filenames)
train_dog = filter(lambda x: x[:3]=='dog', train_filenames)

def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)

rmrf_mkdir('preprocess')
os.mkdir('preprocess/train2')
os.mkdir('preprocess/train2/cat')
os.mkdir('preprocess/train2/dog')

rmrf_mkdir('preprocess/test2')
os.symlink('row_data/test/', 'preprocess/test2/test')

for filename in train_cat:
    os.symlink('row_data/train/'+filename, 'preprocess/train2/cat/'+filename)

for filename in train_dog:
    os.symlink('row_data/train/'+filename, 'preprocess/train2/dog/'+filename)


