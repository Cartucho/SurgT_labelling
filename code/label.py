import glob
import os
from natsort import natsorted
import cv2 as cv
import numpy as np
from pathlib import Path
from code import utils


class Keypoints:
    def __init__(self, dir_out_l, dir_out_r):
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
        # Initialization
        self.im_h = -1
        self.im_w = -1
        self.n_im = len(self.im_path_l)


    def get_n_im(self):
        return self.n_im


    def get_resolution(self):
        return self.im_h, self.im_w


    def get_im_pair(self):
        return self.im_l, self.im_r


    def get_im_pair_name(self, ind_im):
        im_name_l = Path(self.im_path_l[ind_im]).stem
        im_name_r = Path(self.im_path_r[ind_im]).stem
        assert(im_name_l == im_name_r)
        return im_name_l


    def im_update(self, ind_im):
        im_path_l = self.im_path_l[ind_im]
        im_path_r = self.im_path_r[ind_im]
        self.im_l = cv.imread(im_path_l, -1)
        self.im_r = cv.imread(im_path_r, -1)
        if (self.im_h != -1 and self.im_w != -1):
            # Check that images have the same size
            assert(self.im_l.shape[0] == self.im_r.shape[0] == self.im_h)
            assert(self.im_l.shape[1] == self.im_r.shape[1] == self.im_w)
        else:
            self.im_h, self.im_w = self.im_l.shape[:2]


class Draw:
    def __init__(self, config):
        self.ind_im = 0
        self.ind_id = 0
        self.load_data_config(config)
        self.load_vis_config(config)
        self.mouse_u = 0
        self.mouse_v = 0
        self.is_mouse_on_im_l = False
        self.is_mouse_on_im_r = False
        self.initialize_im()


    def load_data_config(self, config):
        c_data = config["data"]
        self.dir_data = c_data["dir"]
        # Images
        dir_l = os.path.join(self.dir_data, c_data["subdir_stereo_l"])
        dir_r = os.path.join(self.dir_data, c_data["subdir_stereo_r"])
        self.is_rectified = c_data["is_rectified"]
        im_format = c_data["im_format"]
        self.Images = Images(dir_l, dir_r, im_format)
        # Keypoints
        dir_out_l = os.path.join(self.dir_data, c_data["subdir_output_l"])
        dir_out_r = os.path.join(self.dir_data, c_data["subdir_output_r"])
        self.Keypoints = Keypoints(dir_out_l, dir_out_r)


    def load_vis_config(self, config):
        c_vis = config["vis"]
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


    def initialize_im(self):
        self.n_im = self.Images.get_n_im()
        self.Images.im_update(self.ind_im)
        self.im_h, self.im_w = self.Images.get_resolution()
        self.update_im_with_keypoints()


    def copy_im_kpt_to_all(self):
        self.im_l_all = np.copy(self.im_l_kpt)
        self.im_r_all = np.copy(self.im_r_kpt)


    def im_draw_guide_line(self):
        self.copy_im_kpt_to_all()
        line_thick = self.guide_t
        color = np.array(self.guide_c, dtype=np.uint8).tolist()
        v = self.mouse_v
        pt_l = (0, v)
        pt_r = (self.im_w, v)
        if self.is_rectified:
            cv.line(self.im_l_all, pt_l, pt_r, color, line_thick)
            cv.line(self.im_r_all, pt_l, pt_r, color, line_thick)
        u = self.mouse_u
        pt_t = (u, 0)
        pt_b = (u, self.im_h)
        if self.is_mouse_on_im_l:
            cv.line(self.im_l_all, pt_t, pt_b, color, line_thick)
            if not self.is_rectified:
                cv.line(self.im_l_all, pt_l, pt_r, color, line_thick)
        elif self.is_mouse_on_im_r:
            cv.line(self.im_r_all, pt_t, pt_b, color, line_thick)
            if not self.is_rectified:
                cv.line(self.im_r_all, pt_l, pt_r, color, line_thick)


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
        self.im_draw_kpt_cross(self.im_l_kpt, kpt_l_u, kpt_l_v, color)
        self.im_draw_kpt_cross(self.im_r_kpt, kpt_r_u, kpt_r_v, color)
        # Draw ind_id


    def im_draw_all_kpts(self):
        kpts_l, kpts_r = self.Keypoints.get_kpts()
        for kpt_l_key, kpt_l_val in kpts_l.items():
            kpt_r_val = kpts_r[kpt_l_key]
            self.im_draw_kpt_pair(kpt_l_key, kpt_l_val, kpt_r_val)


    def mouse_move(self, u, v):
        self.mouse_u = u
        self.mouse_v = v
        # Check if mouse is on left or right image
        self.is_mouse_on_im_l = False
        self.is_mouse_on_im_r = False
        if v < self.im_h:
            if u < self.im_w:
                self.is_mouse_on_im_l = True
            else:
                self.mouse_u -= self.im_w
                self.is_mouse_on_im_r = True
        self.im_draw_guide_line()


    def mouse_lclick(self, u, v):
        self.mouse_u = u
        self.mouse_v = v
        self.update_im_with_keypoints()


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


    def add_status_bar(self, draw):
        # Make black rectangle
        bar = np.zeros((self.bar_h_pxl, draw.shape[1], 3), dtype=draw.dtype)
        # Add text status to bar
        bar = self.add_status_text(bar)
        draw = np.concatenate((draw, bar), axis=0)
        return draw


    def update_im_with_keypoints(self):
        im_l, im_r = self.Images.get_im_pair()
        self.im_l_kpt = np.copy(im_l)
        self.im_r_kpt = np.copy(im_r)
        self.load_kpt_data()
        self.im_draw_all_kpts()
        self.copy_im_kpt_to_all()


    def load_kpt_data(self):
        im_name = self.Images.get_im_pair_name(self.ind_im)
        self.Keypoints.update_ktp_pairs(im_name)


    def im_next(self):
        self.ind_im += 1
        if self.ind_im > (self.n_im - 1):
            self.ind_im = 0
        self.Images.im_update(self.ind_im)
        self.update_im_with_keypoints()


    def im_prev(self):
        self.ind_im -= 1
        if self.ind_im < 0:
            self.ind_im = (self.n_im - 1)
        self.Images.im_update(self.ind_im)
        self.update_im_with_keypoints()


    def id_next(self):
        self.ind_id += 1


    def id_prev(self):
        self.ind_id -=1
        if self.ind_id < 0:
            self.ind_id = 0


    def get_draw(self):
        # Stack images together
        draw = np.concatenate((self.im_l_all, self.im_r_all), axis=1)
        # Add status bar in the bottom
        draw = self.add_status_bar(draw)
        return draw


class Interface:
    def __init__(self, config):
        self.load_keys_config(config)
        self.Draw = Draw(config)
        c_vis = config["vis"]
        self.window_name = c_vis["window_name"]
        self.create_window()


    def load_keys_config(self, config):
        c_keys = config["key"]
        self.key_quit = c_keys["quit"]
        self.key_im_prev = c_keys["im_prev"]
        self.key_im_next = c_keys["im_next"]
        self.key_id_prev = c_keys["id_prev"]
        self.key_id_next = c_keys["id_next"]
        self.key_readjust = c_keys["readjust"]
        self.key_magic = c_keys["magic"]


    def mouse_listener(self, event, x, y, flags, param):
        if (event == cv.EVENT_MOUSEMOVE):
            self.Draw.mouse_move(x, y)
        elif (event == cv.EVENT_LBUTTONUP):
            self.Draw.mouse_lclick(x, y)


    def create_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback(self.window_name, self.mouse_listener)


    def check_key_pressed(self, key_pressed):
        if key_pressed == ord(self.key_im_next):
            self.Draw.im_next()
        elif key_pressed == ord(self.key_im_prev):
            self.Draw.im_prev()
        elif key_pressed == ord(self.key_id_next):
            self.Draw.id_next()
        elif key_pressed == ord(self.key_id_prev):
            self.Draw.id_prev()


    def main_loop(self):
        """ Interface's main loop """
        key_pressed = None
        while key_pressed != ord(self.key_quit):
            draw = self.Draw.get_draw()
            cv.imshow(self.window_name, draw)
            key_pressed = cv.waitKey(1)
            self.check_key_pressed(key_pressed)


def label_data(config):
    inter = Interface(config)
    inter.main_loop()
