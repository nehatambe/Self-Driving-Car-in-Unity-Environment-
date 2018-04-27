import argparse
import base64
import os
import random
import shutil
import traceback
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask

import pos_detect_model
import position
import q_learning_model
from action import get_control_value, get_random_action, get_hardcoded_action
from data_utils import process_image

sio = socketio.Server()
app = Flask(__name__)

FRAME_PER_ACTION = 1
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.5  # starting value of epsilon
EXPLORE = 3000000.  # frames over which to anneal epsilon

t = 0
epsilon = INITIAL_EPSILON

replay_memory = None

qlm = None
pdm = None

control_value = None


@sio.on('telemetry')
def telemetry(sid, data):
    global t, epsilon, control_value, replay_memory
    try:
        if args.image_folder and not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
            replay_memory.reset()
            t = 0
    except Exception as e:
        print(e.with_traceback())
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        if t % 3 == 0:
            try:
                img_orig = Image.open(BytesIO(base64.b64decode(data["image"])))
                img = process_image(img_orig)

                state_img = np.expand_dims(img, 0)
                state_info = np.array([[steering_angle, throttle, speed]])
                state = [state_img, state_info]

                pos_out = pdm.predict(state_img)
                # print(pos_out)
                pos_idx = np.argmax(pos_out, axis=1)[0]
                # print(pos_idx)
                reward = position.compute_reward(pos_idx)

                if args.mode == 'hardcoded':
                    msg = "{:6} Position  {:3} Hardcoded Action: {}->{}"
                    action = get_hardcoded_action(pos_idx)
                elif args.mode == 'drive':
                    msg = "{:6} TESTING   {:3} Predict Action:   {}->{}"
                    action = np.argmax(qlm.predict(state_img), axis=1)[0]
                    if position.get_label(pos_idx) == 'LEFT' and action < 5:
                        action = get_hardcoded_action(pos_idx)
                    elif position.get_label(pos_idx) == 'RIGHT' and action > 3:
                        action = get_hardcoded_action(pos_idx)
                else:
                    if position.get_label(pos_idx) == 'ON':
                        if random.random() <= epsilon:
                            action = get_random_action()
                            msg = "{:6} EXPLORING {:3} Random Action:    {}->{}"
                        else:
                            action = np.argmax(qlm.predict(state_img), axis=1)[0]
                            msg = "{:6} TESTING   {:3} Predict Action:   {}->{}"
                    else:
                        action = get_hardcoded_action(pos_idx)
                        msg = "{:6} REPAIR    {:3} Predict Action:   {}->{}"

                # We reduced the epsilon gradually
                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                if args.mode != 'drive':
                    replay_memory.memorize(reward, img_orig, state, pos_idx, action)
                    replay_memory.store_mini_batch()

                control_value = get_control_value(action, speed)
                print(msg.format(int(t / 3), position.get_label(pos_idx), action, control_value))
            except Exception as e:
                print(e.with_traceback())
        send_control(*control_value)
        t = t + 1
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '-m', '--mode',
        type=str,
        nargs='?',
        default='train',
        help='Run model, can be train, drive, and hardcoded'
    )
    parser.add_argument(
        '-s', '--image-folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    if args.mode not in ['train', 'drive', 'hardcoded']:
        print('Mode must be train, drive, or hardcoded!')
        exit(1)
    if args.image_folder != '':
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)
        print("RECORDING IMAGES in {} ...".format(args.image_folder))
    else:
        print("NOT RECORDING IMAGES ...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)

    replay_memory = q_learning_model.ReplayMemory(args.image_folder)

    pdm = pos_detect_model.build_model()
    if os.path.exists(pos_detect_model.saved_h5_name):
        print('load saved model')
        pdm.load_weights(pos_detect_model.saved_h5_name)

    qlm = q_learning_model.build_model()
    if os.path.exists(q_learning_model.saved_h5_name):
        print('load saved model')
        qlm.load_weights(q_learning_model.saved_h5_name)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
