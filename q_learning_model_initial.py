import json

import os
import tensorflow as tf
from keras import backend as K

from data_utils import load_data
from q_learning_model import build_model, train_model, saved_h5_name, saved_json_name

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

qlm = build_model()

root = '/Volumes/CPSC587DATA'
data_folders = [d for d in os.listdir(root) if d.startswith('RecordedImg2017')]
data_folders.sort()

for folder in data_folders:
    print("Train on", folder)
    info, img_map = load_data(os.path.join(root, folder), check=True)
    train_model(qlm, info, img_map)

    print('Save model...')
    qlm.save_weights(saved_h5_name, overwrite=True)
    with open(saved_json_name, "w") as outfile:
        json.dump(qlm.to_json(), outfile)
