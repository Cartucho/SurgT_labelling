import glob
import os
from natsort import natsorted
import cv2 as cv
import numpy as np




class Interface:
    def __init__(self, config):
        c_data = config["data"]
        self.dir_data = c_data["dir"]
        self.dir_l = os.path.join(self.dir_data, c_data["subdir_stereo_l"])
        self.dir_r = os.path.join(self.dir_data, c_data["subdir_stereo_r"])
        self.im_format = c_data["im_format"]
        # Load keys
        c_keys = config["key"]
        self.key_quit = c_keys["quit"]
        self.key_im_prev = c_keys["im_prev"]
        self.key_im_next = c_keys["im_next"]
        self.key_id_prev = c_keys["id_prev"]
        self.key_id_next = c_keys["id_next"]
        self.key_readjust = c_keys["readjust"]
        self.key_magic = c_keys["magic"]
        # Load visualization data
        c_vis = config["vis"]
        max_w_pxl = c_vis["max_w_pxl"]
        self.im_w_d = int(max_w_pxl / 2)
        # Initialize
        self.ind_im = 0
        self.ind_class = 0
        self.mouse_u = 0
        self.mouse_v = 0

    def load_image_paths(self):
        im_l_path = os.path.join(self.dir_l, "*{}".format(self.im_format))
        self.im_path_l = natsorted(glob.glob(im_l_path))
        im_r_path = os.path.join(self.dir_r, "*{}".format(self.im_format))
        self.im_path_r = natsorted(glob.glob(im_r_path))
        assert(len(self.im_path_l) == len(self.im_path_r))


    def im_resize(self, im_l, im_r):
        im_w_c = im_l.shape[1]
        im_w_d = self.im_w_d
        f = float(im_w_d) / im_w_c # scale factor
        im_l = cv.resize(im_l, None, fx=f, fy=f, interpolation=cv.INTER_AREA)
        im_r = cv.resize(im_r, None, fx=f, fy=f, interpolation=cv.INTER_AREA)
        return im_l, im_r


    def im_augmentation(self, im_l, im_r):
        return im_l, im_r


    def mouse_listener(self, event, x, y, flags, param):
        self.mouse_u = x
        self.mouse_v = y


    def create_window(self, window_name):
        cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback(window_name, self.mouse_listener)


    def im_get(self):
        im_path_l = self.im_path_l[self.ind_im] # TODO: update ind_im carefully
        im_path_r = self.im_path_r[self.ind_im]
        im_l = cv.imread(im_path_l, -1)
        im_r = cv.imread(im_path_r, -1)
        # Check that images have the same size
        assert(im_l.shape == im_r.shape)
        # Create image window
        #cv2.resizeWindow(WINDOW_NAME, 1000, 700)
        # Resize images (from current to desired)
        im_l, im_r = self.im_resize(im_l, im_r)
        # Augment images
        #im_l, im_r = self.im_augmentation(im_l, im_r)
        return im_l, im_r


    def main_loop(self, window_name):
        """ Interface's main loop """
        key_pressed = None
        while key_pressed != ord(self.key_quit):
            im_l, im_r = self.im_get()
            # Stack images together
            stack = np.concatenate((im_l, im_r), axis=1)
            cv.imshow(window_name, stack)
            key_pressed = cv.waitKey(1)
        print("Finished")


def label_data(config):
    inter = Interface(config)
    inter.load_image_paths()
    window_name = "Stereo match labeler"
    inter.create_window(window_name)
    inter.main_loop(window_name)
