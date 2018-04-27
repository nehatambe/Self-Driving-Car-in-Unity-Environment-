import os
import random
import re

import pandas as pd
import shutil

from action import ACTION_NUM

col_names = ['center_img', 'left_img', 'right_img', 'steer_angle', 'throttle', 'speed_n', 'speed']
root = '/Volumes/CPSC587DATA/RecordedImgManual'
csv_path = os.path.join(root, 'driving_log.csv')
log = pd.read_csv(csv_path, header=None, names=col_names, index_col=False)

new_path = '/Volumes/CPSC587DATA/RecordedImg'
if os.path.exists(new_path):
    shutil.rmtree(new_path)
os.makedirs(new_path)


def update_action(row):
    angle = row['steer_angle']
    action = int(round((angle + 1) / 2 * (ACTION_NUM - 1)))
    return action


def update_cur_filename(row):
    filename = re.sub(r'.*2017', '2017', row.center_img)
    shutil.copy(row.center_img, os.path.join(new_path, filename))
    return filename


# print(log.describe())
log['cur_filename'] = log.apply(update_cur_filename, axis=1)
log['cur_pos_idx'] = 1
log['cur_steering_angle'] = log['steer_angle']
log['cur_throttle'] = log['throttle']
log['cur_speed'] = log['speed']
log['action'] = log.apply(update_action, axis=1)
log['reward'] = log.index
log = log.drop(col_names, axis=1)
log['next_filename'] = log['cur_filename'].shift(-1)
log['next_pos_idx'] = 1
log['next_steering_angle'] = log['cur_steering_angle'].shift(-1)
log['next_throttle'] = log['cur_throttle'].shift(-1)
log['next_speed'] = log['cur_speed'].shift(-1)
log = log.drop(log.index[len(log) - 1])
# print(log.head())

frames = []
for i in range(len(log)):
    frames.append(log.sample(n=32))
info = pd.concat(frames)
info.to_csv(os.path.join(new_path, 'info.csv'), index=False)
shutil.rmtree(os.path.join(root, 'IMG'))
os.remove(os.path.join(root, 'driving_log.csv'))
# print(info.describe())
