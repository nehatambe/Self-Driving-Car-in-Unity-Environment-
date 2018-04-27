import json

import os
import tensorflow as tf
from keras import backend as K

from pos_detect_model import build_model, train_model, saved_h5_name, saved_json_name, \
    evaluate_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

K.set_session(sess)

model = build_model()

root = '/Volumes/CPSC587DATA'
data_folders = [d for d in os.listdir(root) if d.startswith('RecordedImg2017')]
data_folders.sort()

for folder in data_folders:
    print("Train on", folder)
    train_model(model, os.path.join(root, folder), n_epoch=100)

    evaluate_model(model, os.path.join(root, folder))
    print('Save model...')
    model.save_weights(saved_h5_name, overwrite=True)
    with open(saved_json_name, "w") as outfile:
        json.dump(model.to_json(), outfile)
