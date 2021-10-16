import glob
import os
from natsort import natsorted
import cv2 as cv
import numpy as np
from pathlib import Path
from code import utils




class ImageKptPairs:
    def __init__(self , dir_out_l, dir_out_r):
        self.kpts_l = {}
        self.kpts_r = {}
        self.dir_out_l = dir_out_l
        self.dir_out_r = dir_out_r
        self.create_output_paths()


    def create_output_paths(self):
        if not os.path.isdir(self.dir_out_l):
            os.mkdir(self.dir_out_l)
        if not os.path.isdir(self.dir_out_r):
            os.mkdir(self.dir_out_r)


    def get_kpts(self):
        return self.kpts_l, self.kpts_r


    def load_kpts_from_file(self, path):
        if not os.path.isfile(path):
            return {}
        # Load data from .yaml file
        return utils.load_yaml_data(path)


    def update_ktp_pairs(self, im_name):
        self.kpts_l = {}
        self.kpts_r = {}
        name_file = "{}.yaml".format(im_name)
        path_l = os.path.join(self.dir_out_l, name_file)
        path_r = os.path.join(self.dir_out_r, name_file)
        self.kpts_l = self.load_kpts_from_file(path_l)
        self.kpts_r = self.load_kpts_from_file(path_r)
        assert(len(self.kpts_l) == len(self.kpts_r))


class Images:
    def __init__(self, dir_l, dir_r, im_format):
        path_l = os.path.join(dir_l, "*{}".format(im_format))
        path_r = os.path.join(dir_r, "*{}".format(im_format))
        self.im_path_l = natsorted(glob.glob(path_l))
        self.im_path_r = natsorted(glob.glob(path_r))
        assert(len(self.im_path_l) == len(self.im_path_r))


    def get_n_im(self):
        return len(self.im_path_l)


    def get_im_pair(self, ind_im):
        im_path_l = self.im_path_l[ind_im]
        im_path_r = self.im_path_r[ind_im]
        im_l = cv.imread(im_path_l, -1)
        im_r = cv.imread(im_path_r, -1)
        return im_l, im_r


    def get_im_pair_name(self, ind_im):
        im_name_l = Path(self.im_path_l[ind_im]).stem
        im_name_r = Path(self.im_path_r[ind_im]).stem
        assert(im_name_l == im_name_r)
        return im_name_l


class Data:
    def __init__(self, config):
        self.load_data_config(config)
        self.Images = Images(self.dir_l, self.dir_r, self.im_format)
        self.KptPairs = ImageKptPairs(self.dir_out_l, self.dir_out_r)


    def load_data_config(self, config):
        c_data = config["data"]
        self.dir_data = c_data["dir"]
        self.dir_l = os.path.join(self.dir_data, c_data["subdir_stereo_l"])
        self.dir_r = os.path.join(self.dir_data, c_data["subdir_stereo_r"])
        self.im_format = c_data["im_format"]
        self.dir_out_l = os.path.join(self.dir_data, c_data["subdir_output_l"])
        self.dir_out_r = os.path.join(self.dir_data, c_data["subdir_output_r"])



class Interface:
    def __init__(self, config):
        self.load_keys_config(config)
        self.load_vis_config(config)
        self.Data = Data(config)
        # Initialize
        self.ind_im = 0
        self.ind_id = 0
        self.mouse_u = 0
        self.mouse_v = 0
        self.n_im = self.Data.Images.get_n_im()
        self.im_h = -1
        self.im_w = -1
        self.im_l = None
        self.im_r = None
        self.im_l_a = None # Augmented images
        self.im_r_a = None # Augmented images


    def load_vis_config(self, config):
        c_vis = config["vis"]
        self.window_name = c_vis["window_name"]
        c_guide = c_vis["guide"]
        self.guide_t = c_guide["thick_pxl"]
        self.guide_c = c_guide["color"]
        c_bar = c_vis["bar"]
        self.bar_h_pxl = c_bar["h_pxl"]
        self.bar_m_l_pxl = c_bar["m_l_pxl"]
        self.bar_text_h_pxl = c_bar["text_h_pxl"]
        self.bar_text_c = c_bar["text_color"]
        c_kpt = c_vis["kpt"]
        self.kpt_c_thick_pxl = c_kpt["c_thick_pxl"]
        self.kpt_c_size_pxl = c_kpt["c_size_pxl"]
        self.kpt_color = c_kpt["color"]
        self.kpt_s_thick_pxl = c_kpt["s_thick_pxl"]


    def load_keys_config(self, config):
        c_keys = config["key"]
        self.key_quit = c_keys["quit"]
        self.key_im_prev = c_keys["im_prev"]
        self.key_im_next = c_keys["im_next"]
        self.key_id_prev = c_keys["id_prev"]
        self.key_id_next = c_keys["id_next"]
        self.key_readjust = c_keys["readjust"]
        self.key_magic = c_keys["magic"]


    def im_draw_guide_line(self):
        line_thick = self.guide_t
        color = np.array(self.guide_c, dtype=np.uint8).tolist()
        v = self.mouse_v
        cv.line(self.im_l_a, (0, v), (self.im_w, v), color, line_thick)
        cv.line(self.im_r_a, (0, v), (self.im_w, v), color, line_thick)
        u = self.mouse_u
        cv.line(self.im_l_a, (u, 0), (u, self.im_h), color, line_thick)
        cv.line(self.im_r_a, (u, 0), (u, self.im_h), color, line_thick)


    def im_draw_kpt_cross(self, im, u, v, color):
        size = self.kpt_c_size_pxl
        # Draw outer square
        s_t = self.kpt_s_thick_pxl
        cv.rectangle(im, (u - size, v - size), (u + size, v + size), color, s_t)
        # Draw inner cross
        c_t = self.kpt_c_thick_pxl
        cv.line(im, (u - size, v), (u + size, v), color, c_t)
        cv.line(im, (u, v - size), (u, v + size), color, c_t)


    def im_draw_kpt_pair(self, ind_id, kpt_l, kpt_r):
        kpt_l_u = kpt_l["u"]
        kpt_l_v = kpt_l["v"]
        kpt_r_u = kpt_r["u"]
        kpt_r_v = kpt_r["v"]
        # Draw cross
        color = np.array(self.kpt_color, dtype=np.uint8).tolist()
        self.im_draw_kpt_cross(self.im_l_a, kpt_l_u, kpt_l_v, color)
        self.im_draw_kpt_cross(self.im_r_a, kpt_r_u, kpt_r_v, color)
        # Draw ind_id


    def im_draw_all_kpts(self):
        kpts_l, kpts_r = self.Data.KptPairs.get_kpts()
        for kpt_l_key, kpt_l_val in kpts_l.items():
            kpt_r_val = kpts_r[kpt_l_key]
            self.im_draw_kpt_pair(kpt_l_key, kpt_l_val, kpt_r_val)


    def im_augmentation(self):
        self.copy_images()
        self.im_draw_guide_line()
        self.im_draw_all_kpts()


    def mouse_listener(self, event, x, y, flags, param):
        self.mouse_u = x
        self.mouse_v = y
        self.im_augmentation()


    def create_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback(self.window_name, self.mouse_listener)


    def get_text_scale_to_fit_height(self, txt, font, thickness):
        _, text_h = cv.getTextSize(txt, font, 1.0, thickness)[0]
        scale = float(self.bar_text_h_pxl) / text_h
        return scale


    def add_status_text(self, bar):
        # Message
        txt = ""
        txt += "Im: [{}/{}]".format(self.ind_im, self.n_im - 1)
        txt += " Id: [{}]".format(self.ind_id)
        # Text specifications
        font = cv.FONT_HERSHEY_DUPLEX
        thickness = 2
        font_scale = self.get_text_scale_to_fit_height(txt, font, thickness)
        color = np.array(self.bar_text_c, dtype=np.uint8).tolist()
        # Centre text vertically
        bot = int((self.bar_h_pxl + self.bar_text_h_pxl) / 2.0)
        left_bot = (self.bar_m_l_pxl, bot) # (left, bottom) corner of the text
        # Write text
        cv.putText(bar, txt, left_bot, font, font_scale, color, thickness)
        return bar


    def add_status_bar(self, stack):
        # Make black rectangle
        bar = np.zeros((self.bar_h_pxl, stack.shape[1], 3), dtype=stack.dtype)
        # Add text status to bar
        bar = self.add_status_text(bar)
        stack = np.concatenate((stack, bar), axis=0)
        return stack


    def im_update(self):
        self.im_l, self.im_r = self.Data.Images.get_im_pair(self.ind_im)
        if (self.im_h != -1 and self.im_w != -1):
            # Check that images have the same size
            assert(self.im_l.shape[0] == self.im_r.shape[0] == self.im_h)
            assert(self.im_l.shape[1] == self.im_r.shape[1] == self.im_w)
        self.copy_images()
        self.load_kpt_data()
        self.im_draw_all_kpts()


    def check_key_pressed(self, key_pressed):
        if key_pressed == ord(self.key_im_next):
            self.ind_im += 1
            if self.ind_im > (self.n_im - 1):
                self.ind_im = 0
            self.im_update()
        elif key_pressed == ord(self.key_im_prev):
            self.ind_im -= 1
            if self.ind_im < 0:
                self.ind_im = (self.n_im - 1)
            self.im_update()
        elif key_pressed == ord(self.key_id_next):
            self.ind_id += 1
        elif key_pressed == ord(self.key_id_prev):
            self.ind_id -=1
            if self.ind_id < 0:
                self.ind_id = 0


    def initialize_im(self):
        self.im_update()
        self.im_h, self.im_w = self.im_l.shape[:2]


    def copy_images(self):
        self.im_l_a = np.copy(self.im_l)
        self.im_r_a = np.copy(self.im_r)


    def load_kpt_data(self):
        im_name = self.Data.Images.get_im_pair_name(self.ind_im)
        self.Data.KptPairs.update_ktp_pairs(im_name)


    def main_loop(self):
        """ Interface's main loop """
        key_pressed = None
        self.initialize_im()
        while key_pressed != ord(self.key_quit):
            # Stack images together
            stack = np.concatenate((self.im_l_a, self.im_r_a), axis=1)
            # Add status bar in the bottom
            stack = self.add_status_bar(stack)
            cv.imshow(self.window_name, stack)
            key_pressed = cv.waitKey(1)
            self.check_key_pressed(key_pressed)


def label_data(config):
    inter = Interface(config)
    inter.create_window()
    inter.main_loop()
