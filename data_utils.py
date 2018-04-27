import os
import tkinter as tk

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import position


class ImageChecker:
    def __init__(self, img_folder) -> None:
        self.csv_path = os.path.join(img_folder, 'info.csv')
        self.info = pd.read_csv(self.csv_path)

        recorded_img_files = set(self.info['cur_filename']).union(set(self.info['next_filename']))
        file_need_remove = [f for f in os.listdir(img_folder)
                            if f not in recorded_img_files and f.endswith('.jpg')]
        if file_need_remove:
            print("Remove files {}".format(file_need_remove))
            for f in file_need_remove:
                os.remove(os.path.join(img_folder, f))

        self.img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
        self.img_files.sort()
        self.img_full_path = [f for f in map(lambda x: os.path.join(img_folder, x), self.img_files)]
        self.index = 0
        self.window = tk.Tk()
        self.window.title("{}/{}".format(self.index + 1, len(self.img_files)))
        self.window.geometry("320x240")
        self.window.configure(background='grey')

        self.img_frame = tk.Frame(self.window, width=320, height=160)
        self.img_frame.pack(side="top")
        img = ImageTk.PhotoImage(Image.open(self.img_full_path[self.index]))
        self.img_panel = tk.Label(self.img_frame, image=img)
        self.img_panel.pack(fill="both", expand="yes")

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(side="bottom", fill="both", expand="yes")
        self.prev_button = tk.Button(self.button_frame, text="Prev", command=self.prev_image)
        self.prev_button.pack(side="left")
        self.next_button = tk.Button(self.button_frame, text="Next", command=self.next_image)
        self.next_button.pack(side="right")

        self.pos_var = tk.IntVar()
        self.pos_var.set(self.get_car_pos_idx(self.img_files[self.index]))
        self.radio_frame = tk.Frame(self.window)
        self.radio_frame.pack(side="top")
        for pos_idx, text in enumerate(position.pos_labels):
            radio_button = tk.Radiobutton(self.radio_frame, text=text, variable=self.pos_var,
                                          value=pos_idx, command=self.pos_select)
            radio_button.pack(side="left")

        self.window.protocol("WM_DELETE_WINDOW", lambda: self.quit())
        self.window.bind('<Escape>', lambda e: self.quit())
        self.window.bind('n', lambda e: self.next_image())
        self.window.bind('p', lambda e: self.prev_image())
        self.window.bind('1', lambda e: self.select_pos(0))
        self.window.bind('2', lambda e: self.select_pos(1))
        self.window.bind('3', lambda e: self.select_pos(2))
        # self.window.bind('4', lambda e: self.select_pos(3))
        self.window.bind('s', lambda e: self.save_csv())
        self.window.mainloop()

    def get_car_pos_idx(self, filename):
        values = self.info.loc[self.info['cur_filename'] == filename, 'cur_pos_idx'].values
        if len(values):
            return values[0]
        else:
            values = self.info.loc[self.info['next_filename'] == filename, 'next_pos_idx'].values
            return values[0] if len(values) else position.get_index('ON')

    def set_car_pos_idx(self, filename, idx):
        self.info.loc[self.info['cur_filename'] == filename, 'cur_pos_idx'] = idx
        self.info.loc[self.info['next_filename'] == filename, 'next_pos_idx'] = idx
        reward = position.compute_reward(idx)
        self.info.loc[self.info['next_filename'] == filename, 'reward'] = reward

    def change_image(self, step):
        self.pos_select()
        self.index = self.index + step
        img = ImageTk.PhotoImage(Image.open(self.img_full_path[self.index]))
        self.img_panel.configure(image=img)
        self.img_panel.image = img
        rewards_values = self.info.loc[
            self.info['next_filename'] == self.img_files[self.index], 'reward'].values
        reward = rewards_values[0] if len(rewards_values) else 0
        self.window.title("{}/{} Rewards: {}".format(self.index + 1, len(self.img_files), reward))
        self.pos_var.set(self.get_car_pos_idx(self.img_files[self.index]))

    def next_image(self):
        if self.index < len(self.img_files) - 1:
            self.change_image(1)

    def prev_image(self):
        if self.index > 0:
            self.change_image(-1)

    def pos_select(self):
        self.set_car_pos_idx(self.img_files[self.index], self.pos_var.get())

    def select_pos(self, idx):
        self.pos_var.set(idx)
        self.pos_select()
        self.next_image()

    def save_csv(self):
        print("save info to csv")
        self.info.to_csv(self.csv_path, index=False)

    def quit(self):
        self.save_csv()
        self.window.destroy()


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def process_image(img):
    image = np.asarray(img)
    image = cv2.resize(image, (160, 320), cv2.INTER_AREA)
    image = image[60:-25, :, :]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # print(image.shape)
    return image


def load_data(img_folder, check=False):
    if check:
        checker = ImageChecker(img_folder)
        info = checker.info
    else:
        csv_path = os.path.join(img_folder, 'info.csv')
        info = pd.read_csv(csv_path)
    img_map = {}
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    for f in img_files:
        img = Image.open(os.path.join(img_folder, f))
        img = process_image(img)
        img_map[f] = np.expand_dims(img, 0)
    return info, img_map


def update_rewards(data_folder):
    def update(row):
        return position.compute_reward(row['next_pos_idx'])

    csv_path = os.path.join(data_folder, 'info.csv')
    info = pd.read_csv(csv_path)
    info['reward'] = info.apply(update, axis=1)
    info.to_csv(csv_path)


if __name__ == '__main__':
    root = '/Volumes/CPSC587DATA'
    data_folders = [d for d in os.listdir(root) if d.startswith('RecordedImg2017')]
    data_folders.sort()
    for folder in data_folders:
        print(folder)
        folder = os.path.join(root, folder)
        load_data(folder, True)
        # csv_path = os.path.join(folder, 'info.csv')
        # info = pd.read_csv(csv_path)
        # cols = [c for c in info.columns if not c.startswith("Unnamed")]
        # info.to_csv(csv_path + ".bak", index=False)
        # info[cols].to_csv(csv_path, index=False)
    pass
