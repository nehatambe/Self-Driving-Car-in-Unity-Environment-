import numpy as np
import pandas as pd
from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, \
    recall_score

from data_utils import load_data, IMAGE_SHAPE
from position import POS_NUM

LEARNING_RATE = 1e-4

saved_h5_name = 'pos_detect_model.h5'
saved_json_name = 'pos_detect_model.json'


def build_model(learn_rate=LEARNING_RATE):
    print("Now we build the Position Detect model")
    inputs = Input(shape=IMAGE_SHAPE)
    values = Lambda(lambda x: x / 127.5 - 1.0)(inputs)

    conv_layer_1 = Conv2D(24, (5, 5), strides=(2, 2), padding='same', activation='relu')
    values = conv_layer_1(values)

    conv_layer_2 = Conv2D(36, (5, 5), strides=(2, 2), padding='same', activation='relu')
    values = conv_layer_2(values)

    conv_layer_3 = Conv2D(48, (5, 5), strides=(2, 2), padding='same', activation='relu')
    values = conv_layer_3(values)

    conv_layer_4 = Conv2D(64, (3, 3), padding='same', activation='relu')
    values = conv_layer_4(values)

    conv_layer_5 = Conv2D(64, (3, 3), padding='same', activation='relu')
    values = conv_layer_5(values)

    # values = dropout(values, 0.5)
    values = Dropout(0.5)(values)

    values = Flatten()(values)

    values = Dense(100, activation='relu')(values)
    values = Dense(50, activation='relu')(values)
    values = Dense(10, activation='relu')(values)

    outputs = Dense(POS_NUM)(values)

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=adam)

    print("We finish building the Position Detect model")
    # model.summary()
    return model


def train_model(model, info, img_map, n_epoch=10):
    column_map = {'cur_filename': 'filename',
                  'next_filename': 'filename',
                  'cur_pos_idx': 'pos_idx',
                  'next_pos_idx': 'pos_idx'}
    info = pd.concat([info[['cur_filename', 'cur_pos_idx']].rename(columns=column_map),
                      info[['next_filename', 'next_pos_idx']].rename(columns=column_map)],
                     ignore_index=True)
    info = info.drop_duplicates()
    input_img = np.concatenate([i for i in map(lambda x: img_map[x], info['filename'])], axis=0)
    output_pos = to_categorical(info['pos_idx'], num_classes=POS_NUM)
    for epoch in range(n_epoch):
        loss = model.train_on_batch(input_img, output_pos)
        print("Epoch:{:4} Loss: {:.6f}".format(epoch, loss))


def evaluate_model(model, data_folder, check=False):
    info, img_map = load_data(data_folder, check)
    column_map = {'cur_filename': 'filename',
                  'next_filename': 'filename',
                  'cur_pos_idx': 'pos_idx',
                  'next_pos_idx': 'pos_idx'}
    info = pd.concat([info[['cur_filename', 'cur_pos_idx']].rename(columns=column_map),
                      info[['next_filename', 'next_pos_idx']].rename(columns=column_map)],
                     ignore_index=True)
    info = info.drop_duplicates()
    y_true = info['pos_idx'].values

    input_img = np.concatenate([i for i in map(lambda x: img_map[x], info['filename'])], axis=0)
    output_pos = model.predict(input_img)
    y_pred = np.argmax(output_pos, axis=1)

    print(confusion_matrix(y_true, y_pred))
    print('{:15}'.format('Accuracy:'), accuracy_score(y_true, y_pred))
    print('{:15}'.format('Precision:'), precision_score(y_true, y_pred, average=None))
    print('{:15}'.format('Recall:'), recall_score(y_true, y_pred, average=None))
    print('{:15}'.format('f1 score:'), f1_score(y_true, y_pred, average=None))


if __name__ == '__main__':
    build_model()
