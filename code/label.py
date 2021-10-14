import glob
import os
from natsort import natsorted


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


    def load_image_paths(self):
        im_l_path = os.path.join(self.dir_l, "*{}".format(self.im_format))
        self.im_path_l = natsorted(glob.glob(im_l_path))
        im_r_path = os.path.join(self.dir_r, "*{}".format(self.im_format))
        self.im_path_r = natsorted(glob.glob(im_r_path))


    def main_loop(self):
        """ Interface's main loop """
        key_pressed = None
        while key_pressed != ord(self.key_quit):
            key_pressed = cv.waitKey(20)
        print("Finished")


def label_data(config):
    inter = Interface(config)
    inter.load_image_paths()
    inter.main_loop()
