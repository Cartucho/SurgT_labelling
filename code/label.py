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
        self.dir_out_l = os.path.join(self.dir_data, c_data["subdir_output_l"])
        self.dir_out_r = os.path.join(self.dir_data, c_data["subdir_output_r"])
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
        self.window_name = c_vis["window_name"]
        c_guide = c_vis["guide"]
        self.guide_t = c_guide["thick_pxl"]
        self.guide_c = c_guide["color"]
        c_bar = c_vis["bar"]
        self.h_pxl = c_bar["h_pxl"]
        self.text_h_pxl = c_bar["text_h_pxl"]
        self.text_c = c_bar["text_color"]
        # Initialize
        self.ind_im = 0
        self.ind_id = 0
        self.mouse_u = 0
        self.mouse_v = 0
        self.n_im = -1
        self.im_h = -1
        self.im_w = -1


    def create_output_paths(self):
        if not os.path.isdir(self.dir_out_l):
            os.mkdir(self.dir_out_l)
        if not os.path.isdir(self.dir_out_r):
            os.mkdir(self.dir_out_r)


    def load_image_paths(self):
        im_l_path = os.path.join(self.dir_l, "*{}".format(self.im_format))
        self.im_path_l = natsorted(glob.glob(im_l_path))
        im_r_path = os.path.join(self.dir_r, "*{}".format(self.im_format))
        self.im_path_r = natsorted(glob.glob(im_r_path))
        assert(len(self.im_path_l) == len(self.im_path_r))
        self.n_im = len(self.im_path_l)


    def im_draw_guide_line(self, im_l, im_r):
        line_thick = self.guide_t
        color = np.array(self.guide_c, dtype=np.uint8).tolist()
        v = self.mouse_v
        im_l = cv.line(im_l, (0, v), (self.im_w, v), color, line_thick)
        im_r = cv.line(im_r, (0, v), (self.im_w, v), color, line_thick)
        u = self.mouse_u
        im_l = cv.line(im_l, (u, 0), (u, self.im_h), color, line_thick)
        im_r = cv.line(im_r, (u, 0), (u, self.im_h), color, line_thick)
        return im_l, im_r


    def im_augmentation(self, im_l, im_r):
        self.im_draw_guide_line(im_l, im_r)
        return im_l, im_r


    def mouse_listener(self, event, x, y, flags, param):
        self.mouse_u = x
        self.mouse_v = y


    def create_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback(self.window_name, self.mouse_listener)


    def im_get(self):
        im_path_l = self.im_path_l[self.ind_im]
        im_path_r = self.im_path_r[self.ind_im]
        im_l = cv.imread(im_path_l, -1)
        im_r = cv.imread(im_path_r, -1)
        # Check that images have the same size
        if (self.im_h != -1 and self.im_w != -1):
            assert(im_l.shape[0] == im_r.shape[0] == self.im_h)
            assert(im_l.shape[1] == im_r.shape[1] == self.im_w)
        # Augment images
        im_l, im_r = self.im_augmentation(im_l, im_r)
        return im_l, im_r


    def get_text_scale_to_fit_height(self, txt, font, thickness):
        _, text_h = cv.getTextSize(txt, font, 1.0, thickness)[0]
        scale = float(self.text_h_pxl) / text_h
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
        color = np.array(self.text_c, dtype=np.uint8).tolist()
        # Centre text vertically
        bot = int((self.h_pxl + self.text_h_pxl) / 2.0)
        left_bot = (0, bot) # (left, bottom) corner of the text
        # Write text
        cv.putText(bar, txt, left_bot, font, font_scale, color, thickness)
        return bar


    def add_status_bar(self, stack):
        # Make black rectangle
        bar = np.zeros((self.h_pxl, stack.shape[1], 3), dtype=stack.dtype)
        # Add text status to bar
        bar = self.add_status_text(bar)
        stack = np.concatenate((stack, bar), axis=0)
        return stack


    def check_key_pressed(self, key_pressed):
        if key_pressed == ord(self.key_im_next):
            self.ind_im += 1
            if self.ind_im > (self.n_im - 1):
                self.ind_im = 0
        elif key_pressed == ord(self.key_im_prev):
            self.ind_im -= 1
            if self.ind_im < 0:
                self.ind_im = (self.n_im - 1)
        elif key_pressed == ord(self.key_id_next):
            self.ind_id += 1
        elif key_pressed == ord(self.key_id_prev):
            self.ind_id -=1
            if self.ind_id < 0:
                self.ind_id = 0


    def update_im_resolution(self):
        im_l, _ = self.im_get()
        self.im_h, self.im_w = im_l.shape[:2]


    def main_loop(self, window_name):
        """ Interface's main loop """
        key_pressed = None
        self.update_im_resolution()
        while key_pressed != ord(self.key_quit):
            im_l, im_r = self.im_get()
            # Stack images together
            stack = np.concatenate((im_l, im_r), axis=1)
            # Add status bar in the bottom
            stack = self.add_status_bar(stack)
            cv.imshow(self.window_name, stack)
            key_pressed = cv.waitKey(1)
            self.check_key_pressed(key_pressed)
        print("Finished")


def label_data(config):
    inter = Interface(config)
    inter.create_output_paths()
    inter.load_image_paths()
    inter.create_window()
    inter.main_loop()
