import random

import numpy as np

import position

ACTION_NUM = 9

steering_angle_values = np.linspace(-1, 1, ACTION_NUM)
hardcoded_action = {
    'LEFT': 6,
    'RIGHT': 2,
    'ON': 4
}


def create_action(steering_angle):
    def f(x, n): return int(round((x + 1) / 2 * (n - 1)))

    return f(steering_angle, ACTION_NUM)


def get_control_value(action, speed):
    steering_angle = steering_angle_values[int(action)]
    throttle = (25 - speed) / 25
    return steering_angle, throttle


def get_random_action():
    return random.randint(2, ACTION_NUM - 3)


def get_hardcoded_action(pos_idx):
    return hardcoded_action[position.get_label(pos_idx)]


if __name__ == '__main__':
    for a in range(ACTION_NUM):
        for s in range(30):
            print(a, get_control_value(a, s))
