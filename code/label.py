import glob
import os
class Interface:
    def __init__(self, config):
        c_data = config["data"]
        self.dir_data = c_data["dir"]
        self.dir_l = os.path.join(self.dir_data, c_data["subdir_stereo_l"])
        self.dir_r = os.path.join(self.dir_data, c_data["subdir_stereo_r"])
        self.im_format = c_data["im_format"]


    def load_image_paths(self):
        im_l_path = os.path.join(self.dir_l, "*{}".format(self.im_format))
        self.im_path_l = glob.glob(im_l_path)
        im_r_path = os.path.join(self.dir_r, "*{}".format(self.im_format))
        self.im_path_r = glob.glob(im_r_path)
        print(self.im_path_l)


    def main_loop(self):
        """ Interface's main loop """
        print("Finished")


def label_data(config):
    inter = Interface(config)
    inter.load_image_paths()
    inter.main_loop()
