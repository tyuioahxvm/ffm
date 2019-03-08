# -*- coding:utf-8 -*-

import xlearn as xl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Train

## Task: like

ffm_model_like = xl.create_ffm()
ffm_model_like.setTrain('./train_valid/train_like.ffm')
ffm_model_like.setValidate('./train_valid/valid_like.ffm')

# set params
param = {'task':'binary', 'lr':0.002, 'lambda': 0.002, 'metric':'acc'}

# train
ffm_model_like.fit(param, './saved_models/model_like.out')


# Validation

# get label
label_like_train = [float(line[0]) for line in open('train_valid/train_like.ffm')]
label_like_valid = [float(line[0]) for line in open('train_valid/valid_like.ffm')]

ffm_model = xl.create_ffm()

# prediction task
ffm_model.setTest("./train_valid/valid_like.ffm")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1
ffm_model.predict('./saved_models/model.out', './results/valid_like.txt')

predict = pd.read_csv('./results/valid_like.txt', header=None).values.reshape(-1)

acc = accuracy_score(label_like_valid, predict)

print('accuary of valid: {}'.format(acc))