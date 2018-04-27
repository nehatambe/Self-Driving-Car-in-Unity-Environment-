import csv
import os
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, Flatten, Dense, Lambda, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

import position
from action import ACTION_NUM
from data_utils import IMAGE_SHAPE
from position import POS_NUM

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99

saved_h5_name = 'q_learning_model.h5'
saved_json_name = 'q_learning_model.json'


class ReplayMemory:
    def __init__(self, store_folder, max_num=50000) -> None:
        self.csv_fieldnames = ['cur_filename', 'cur_pos_idx',
                               'cur_steering_angle', 'cur_throttle', 'cur_speed',
                               'action', 'reward', 'next_filename', 'next_pos_idx',
                               'next_steering_angle', 'next_throttle', 'next_speed']
        self.memory = deque()
        self.max_num = max_num
        self.store_folder = store_folder
        if store_folder:
            self.csv_path = os.path.join(store_folder, 'info.csv')
            with open(self.csv_path, 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
                writer.writeheader()
        self.thread_pool = ThreadPoolExecutor(4)

        self.state = None
        self.filename = None
        self.pos_idx = None
        self.action = None
        self.reward = None
        self.next_state = None

    def __len__(self) -> int:
        return len(self.memory)

    def reset(self):
        self.memory.clear()
        with open(self.csv_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
            writer.writeheader()

    def memorize(self, reward, img_orig, state, pos_idx, action) -> None:
        # print(state)
        img_filename = self.save_img(img_orig)
        self.reward = reward
        self.next_state = state
        if self.state is not None and self.action is not None:
            cur_state_info = np.squeeze(self.state[1])
            next_state_info = np.squeeze(self.next_state[1])
            self.memory.append(
                {"cur_filename": self.filename,
                 # "cur_state": self.state,
                 "cur_pos_idx": self.pos_idx,
                 'cur_steering_angle': cur_state_info[0],
                 'cur_throttle': cur_state_info[1],
                 'cur_speed': cur_state_info[2],
                 "action": self.action,
                 "reward": self.reward,
                 "next_filename": img_filename,
                 "next_pos_idx": pos_idx,
                 "next_steering_angle": next_state_info[0],
                 'next_throttle': next_state_info[1],
                 'next_speed': next_state_info[2],
                 # "next_state": self.next_state,
                 })
            if len(self.memory) > self.max_num:
                self.memory.popleft()
        self.filename = img_filename
        self.state = state
        self.pos_idx = pos_idx
        self.action = action

    def get_mini_batch(self, k=BATCH_SIZE):
        return random.sample(self.memory, k) if len(self.memory) > k * 4 else None

    def store_mini_batch(self, k=BATCH_SIZE):
        samples = random.sample(self.memory, k) if len(self.memory) > k * 4 else None
        if samples is not None:
            with open(self.csv_path, 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_fieldnames)
                csv_writer.writerows(samples)

    def save_img(self, img_orig):
        if self.store_folder:
            filename = '{}.jpg'.format(datetime.now().strftime('%Y%m%d%H%M%S.%f')[:-3])
            full_path = os.path.join(self.store_folder, filename)
            self.thread_pool.submit(lambda: img_orig.save(full_path))
            return filename
        else:
            return None


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

    outputs = Dense(ACTION_NUM)(values)

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=adam)

    print("We finish building the Position Detect model")
    # model.summary()
    return model


def train_model(model, info, img_map, batch_size=BATCH_SIZE):
    n_rows = info.shape[0]
    for start in range(0, n_rows, batch_size):
        end = start + batch_size
        if end > n_rows:
            end = n_rows
        sl = slice(start, end)
        file_names = info['cur_filename'][sl]
        inputs = np.concatenate([i for i in map(lambda x: img_map[x], file_names)], axis=0)
        outputs = model.predict(inputs)
        next_inputs = [i for i in map(lambda x: img_map[x], info['next_filename'][sl])]
        for index, row in info[['cur_pos_idx', 'action', 'reward']][sl].iterrows():
            index = index - start
            action = int(row['action'])
            pos = position.get_label(int(row['cur_pos_idx']))
            reward = row['reward']
            orig = outputs[index, action]
            if pos == 'ON':
                next_state = next_inputs[index]
                q_sa = model.predict(next_state)
                outputs[index, action] = reward + GAMMA * np.max(q_sa)
            else:
                outputs[index, action] = reward
            # if index % 8 == 0:
            #     print("\tAction {}, {} -> {}".format(action, orig, output[index, action]))
        loss = model.train_on_batch(inputs, outputs)
        print("Iter:{:4} Loss: {:.6f}".format(int(start / batch_size), loss))
