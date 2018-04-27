import json
import os
import shutil
from datetime import datetime

import tensorflow as tf
from keras import backend as K

import pos_detect_model
import q_learning_model
from data_utils import load_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)


def load_model(model, filename):
    if os.path.exists(filename):
        print('load ', filename)
        model.load_weights(filename)


def save_model(model, filename):
    if os.path.exists(filename):
        bak_name = '{}-{}'.format(filename, datetime.now().strftime('%Y%m%d%H%M%S'))
        shutil.move(filename, bak_name)
    model.save_weights(filename, overwrite=True)
    print('save ', filename)


record_folder = '/Volumes/CPSC587DATA/RecordedImg'
info, img_map = load_data(record_folder, check=True)

pdm = pos_detect_model.build_model()
load_model(pdm, pos_detect_model.saved_h5_name)
pos_detect_model.train_model(pdm, info, img_map, n_epoch=10)
pos_detect_model.evaluate_model(pdm, record_folder)
save_model(pdm, pos_detect_model.saved_h5_name)

qlm = q_learning_model.build_model()
load_model(qlm, q_learning_model.saved_h5_name)
q_learning_model.train_model(qlm, info, img_map)
save_model(qlm, q_learning_model.saved_h5_name)

new_folder = record_folder + datetime.now().strftime('%Y%m%d%H%M%S')
print('move {} to {}'.format(record_folder, new_folder))
shutil.move(record_folder, new_folder)
