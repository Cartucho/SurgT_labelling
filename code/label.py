import glob
import os
import math

from natsort import natsorted
import cv2 as cv
import numpy as np
from pathlib import Path
from code import utils
from scipy.interpolate import interp1d


class Keypoints:
    def __init__(self, dir_out_l, dir_out_r):
        self.kpts_l = {}
        self.kpts_r = {}
        self.dir_out_l = dir_out_l
        self.dir_out_r = dir_out_r
        self.create_output_paths()
        self.path_l = None
        self.path_r = None
        self.new_l = None
        self.new_r = None


    def add_kpt_pair(self, ind_id, kpt_l, kpt_r):
        self.kpts_l[ind_id] = kpt_l
        self.kpts_r[ind_id] = kpt_r
        self.save_kpt_pairs_to_files()


    def new_kpt(self, is_l_kpt, ind_id, u, v):
        kpt_n = {"u": u,
                 "v": v,
                 "is_interp": False,
                 "is_visible": True,
                 "is_difficult": False}
        if is_l_kpt:
            self.new_l = kpt_n
            self.kpts_l[ind_id] = self.new_l
        else:
            self.new_r = kpt_n
            self.kpts_r[ind_id] = self.new_r
        self.check_for_new_kpt_pair()


    def new_intrp_pair(self, ind_id, u_l, v_l, u_r, v_r):
        k_l = {"u": u_l,
               "v": v_l,
               "is_interp": True,
               "is_visible": True,
               "is_difficult": False}
        k_r = {"u": u_r,
               "v": v_r,
               "is_interp": True,
               "is_visible": True,
               "is_difficult": False}
        self.add_kpt_pair(ind_id, k_l, k_r)


    def get_new_kpt_l(self):
        return self.new_l


    def get_new_kpt_r(self):
        return self.new_r


    def check_positive_disparity(self):
        u_l = self.new_l["u"]
        u_r = self.new_r["u"]
        disp = u_l - u_r
        if disp > 0:
            return True
        else:
            print("Error: disparity should be positive!")
            exit()


    def check_for_new_kpt_pair(self):
        if self.new_l is not None and \
           self.new_r is not None:
            if self.check_positive_disparity():
                self.save_kpt_pairs_to_files()
            self.new_l = None
            self.new_r = None


    def create_output_paths(self):
        if not os.path.isdir(self.dir_out_l):
            os.mkdir(self.dir_out_l)
        if not os.path.isdir(self.dir_out_r):
            os.mkdir(self.dir_out_r)


    def eliminate_unpaired_kpts(self):
        keys_l = self.kpts_l.keys()
        keys_r = self.kpts_r.keys()
        not_pair = list(set(keys_l).symmetric_difference(keys_r))
        for key in not_pair:
            self.kpts_l.pop(key, None)
            self.kpts_r.pop(key, None)
            self.new_l = None
            self.new_r = None


    def save_kpt_pairs_to_files(self):
        self.eliminate_unpaired_kpts()
        utils.write_yaml_data(self.path_l, self.kpts_l)
        utils.write_yaml_data(self.path_r, self.kpts_r)


    def eliminate_kpts(self, ind_id):
        self.kpts_l.pop(ind_id, None)
        self.kpts_r.pop(ind_id, None)
        # Save kpts to .yaml
        self.save_kpt_pairs_to_files()
        # Reset new_kpt
        self.new_l = None
        self.new_r = None


    def get_kpts(self):
        return self.kpts_l, self.kpts_r


    def get_kpts_given_ind_id(self, ind_id):
        return self.kpts_l.get(ind_id), self.kpts_r.get(ind_id)


    def load_kpts_from_file(self, path):
        if os.path.isfile(path):
            # Load data from .yaml file
            data = utils.load_yaml_data(path)
            if data is not None:
                return data
        return {}


    def update_ktp_pairs(self, im_name):
        self.kpts_l = {}
        self.kpts_r = {}
        name_file = "{}.yaml".format(im_name)
        self.path_l = os.path.join(self.dir_out_l, name_file)
        self.path_r = os.path.join(self.dir_out_r, name_file)
        self.kpts_l = self.load_kpts_from_file(self.path_l)
        self.kpts_r = self.load_kpts_from_file(self.path_r)
        assert(len(self.kpts_l) == len(self.kpts_r))


    def toggle_is_visibile(self, ind_id):
        """
         Cases:
         1. No kpt labelled

                If kpt is None, then set `is_visible`=False

         2. Kpt labelled

                If `is_visible`=True, then set `is_visible`=False

         3. Picture already marked as not visible

                If `is_visible`=False, then delete the kpts to
                relabel them later, which will set `is_visible`=True
        """
        kpt_l, kpt_r = self.get_kpts_given_ind_id(ind_id)
        # Case 3.
        if kpt_l is not None and kpt_r is not None:
            if not kpt_l["is_visible"] and not kpt_r["is_visible"]:
                self.eliminate_kpts(ind_id)
                return
        # Case 1. and 2.
        kpt_not_vis = {"is_visible": False}
        self.kpts_l[ind_id] = kpt_not_vis
        self.kpts_r[ind_id] = kpt_not_vis
        self.save_kpt_pairs_to_files()


    def toggle_is_difficult(self, ind_id):
        """
            You can only mark as `is_difficult` the keypoints
             that are already labelled.
        """
        kpt_l, kpt_r = self.get_kpts_given_ind_id(ind_id)
        if kpt_l is not None and kpt_r is not None:
            new_bool = not (kpt_l["is_difficult"])
            kpt_l["is_difficult"] = new_bool
            kpt_r["is_difficult"] = new_bool
            self.save_kpt_pairs_to_files()


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


class Interpolation:
    def __init__(self, Images, Keypoints):
        self.Images = Images
        self.Keypoints = Keypoints


    def get_interp_values(self, im_an, an_loc, i_min, i_max, i_n):
        inds_im = np.linspace(i_min, i_max, num=i_n, endpoint=True)
        if len(im_an) > 3:
            f = interp1d(im_an, an_loc, kind='cubic')
        else:
            f = interp1d(im_an, an_loc, kind='linear')
        interp_values = f(inds_im)
        return np.rint(interp_values)


    def get_kpt_anc_in_range(self, rng, ind_id, data_kpt_intrp):
        """ Get anchors for interpolation """
        for i in rng:
            im_name = self.Images.get_im_pair_name(i)
            self.Keypoints.update_ktp_pairs(im_name)
            k_l, k_r = self.Keypoints.get_kpts_given_ind_id(ind_id)
            if k_l is None or k_r is None:
                data_kpt_intrp[im_name] = None
                continue
            if not k_l["is_visible"] or not k_r["is_visible"]:
                break
            data_kpt_intrp[im_name] = None
            if not k_l["is_interp"] and not k_r["is_interp"]:
                data_kpt_intrp[im_name] = {"k_l": k_l, "k_r": k_r}
        return data_kpt_intrp


    def get_kpt_data_given_id_and_im(self, ind_id, ind_im):
        """ 1. Go through all picture-pairs and get all kpts
                with `ind_id` = self.ind_id.

                Get only the ones that were manually labelled,
                since those are the accurate positions that are used
                for interpolation.

                Get only visible kpts that are connected to ind_im.
        """
        data_kpt_intrp = {}
        # Get data before `ind_im`
        rng = range(ind_im, -1, -1) # From `ind_im` to 0, since exclusive
        data_kpt_intrp = self.get_kpt_anc_in_range(rng, ind_id, data_kpt_intrp)
        # Get data after `ind_im`
        rng = range(ind_im + 1, self.Images.n_im, 1)
        data_kpt_intrp = self.get_kpt_anc_in_range(rng, ind_id, data_kpt_intrp)
        # Count non-Nones
        n_non_nones = sum(x is not None for x in data_kpt_intrp.values())
        if n_non_nones < 2: # Need at least 2 points to interpolate
            return None
        return data_kpt_intrp


    def interp_kpts_and_save(self, data_kpt_intrp, ind_id):
        """ Interpolate in between frames """
        im_an = [] # Anchors used for interpolation
        kpt_l_u = []
        kpt_l_v = []
        kpt_r_u = []
        kpt_r_v = []
        for i, (k_key, k_data) in enumerate(natsorted(data_kpt_intrp.items())):
            if k_data is not None:
                im_an.append(i)
                kpt_l_u.append(k_data["k_l"]["u"])
                kpt_l_v.append(k_data["k_l"]["v"])
                kpt_r_u.append(k_data["k_r"]["u"])
                kpt_r_v.append(k_data["k_r"]["v"])
        i_min = im_an[0]
        i_max = im_an[-1]
        i_n = i_max - i_min + 1 # Number of images to interpolate
        interp_k_l_u = self.get_interp_values(im_an, kpt_l_u, i_min, i_max, i_n)
        interp_k_l_v = self.get_interp_values(im_an, kpt_l_v, i_min, i_max, i_n)
        interp_k_r_u = self.get_interp_values(im_an, kpt_r_u, i_min, i_max, i_n)
        interp_k_r_v = interp_k_l_v # Since images are rectified
        """ Save interpolation data """
        im_h, im_w = self.Images.get_resolution()
        for i, (k_key, k_val) in enumerate(natsorted(data_kpt_intrp.items())):
            if k_val is None: # If not an anchor
                if i > i_min and i < i_max: # If inside the interpolated range
                    # Replace None by the interpolated value
                    ind = i - i_min
                    u_l = int(interp_k_l_u[ind])
                    v_l = int(interp_k_l_v[ind])
                    u_r = int(interp_k_r_u[ind])
                    v_r = int(interp_k_r_v[ind])
                    if u_l < 0 or u_r < 0:
                        continue
                    if u_l > im_w or u_r > im_w:
                        continue
                    if v_l > im_h or v_r > im_h:
                        continue
                    if v_l < 0 or v_r < 0:
                        continue
                    self.Keypoints.update_ktp_pairs(k_key)
                    self.Keypoints.new_intrp_pair(ind_id, u_l, v_l, u_r, v_r)


    def start(self, ind_id, ind_im):
        data_kpt_intrp = self.get_kpt_data_given_id_and_im(ind_id, ind_im)
        if data_kpt_intrp is None:
            return
        self.interp_kpts_and_save(data_kpt_intrp, ind_id)


class GT:
    def __init__(self, v, Images, Keypoints, radius, file_out):
        self.video = v
        self.Images = Images
        self.Keypoints = Keypoints
        self.file_out = file_out
        self.radius = radius
        self.baseline = 1. / self.video.Q[3, 2]
        self.P1 = self.video.P1
        self.P1_transp = np.transpose(self.P1)
        self.P2 = self.video.P2
        self.P2_transp = np.transpose(self.P2)
        # Note: `self.Q` != `self.video.Q`, the sphere's conic is diff from the rectification Q
        self.Q = np.array([[1., 0., 0.,  0.],
                           [0., 1., 0.,  0.],
                           [0., 0., 1.,  0.],
                           [0., 0., 0., -radius**2]])
        self.Q /= self.Q[3,3]
        self.H_inv = np.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.],
                               [0., 0., 0.]
                               ])


    def get_kpt_3d_pt(self, k_l, k_r):
        disp = k_l["u"] - k_r["u"]
        pt_2d = np.array([[k_l["u"]],
                          [k_l["v"]],
                          [disp],
                          [1.0]
                          ], dtype=np.float32)
        pt_3d = np.matmul(self.video.Q, pt_2d)
        assert(disp > 0)
        pt_3d /= pt_3d[3, 0]
        return pt_3d


    def get_ellipse_param(self, P, Q_, P_transp):
        C = np.linalg.inv(P @ np.linalg.inv(Q_) @ P_transp)
        #C = C / C[2,2]

        Q = C # Renaming C to Q to make formulas consistent with Wikipedia
        # Get the coefficients
        A = Q[0, 0]
        B = Q[0, 1] * 2.0
        C = Q[1, 1]
        D = Q[0, 2] * 2.0
        E = Q[1, 2] * 2.0
        F = Q[2, 2]

        # Check if it is indeed an ellipse
        if np.linalg.det(Q) == 0:
            raise ValueError("Degenerate conic found!")

        if np.linalg.det(Q[:2,:2]) <= 0: # According to Wikipedia
            raise ValueError("These parameters do not define an ellipse!")

        # Get centre
        denominator = B**2 - 4*A*C
        centre_x = (2*C*D - B*E) / denominator
        centre_y = (2*A*E - B*D) / denominator
        #print("Centre x:{} y:{}".format(centre_x, centre_y))

        # Get major and minor axes
        K = - np.linalg.det(Q[:3,:3]) / np.linalg.det(Q[:2,:2])
        root = math.sqrt(((A - C)**2 + B**2))
        a = math.sqrt(2*K / (A + C - root))
        b = math.sqrt(2*K / (A + C + root))
        #print("Major:{} minor:{}".format(a, b))

        # Get angle
        angle = math.atan2(C - A + root, B)
        angle *= 180.0/math.pi # Convert angle to degrees
        #print("Angle in degrees: {}".format(angle))
        return (centre_x, centre_y, a, b, angle)


    def get_bbox_from_ellipse(self, ellipse_info):
        mask = np.zeros((self.Images.im_h, self.Images.im_w), dtype=np.uint8)
        centre_x, centre_y, a, b, angle = ellipse_info
        mask = cv.ellipse(mask,
                          (int(round(centre_x)), int(round(centre_y))),
                          (int(round(b)), int(round(a))),
                          angle,
                          startAngle=0,
                          endAngle=360,
                          color=(255),
                          thickness=1,
                          #lineType=cv.LINE_AA
                         )
        # Get bbox around contour
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        bbox = cv.boundingRect(contours[0])
        """
        # Uncomment to visualize the ellipse and rectangle
        cv.rectangle(mask,
                     (int(bbox[0]), int(bbox[1])),
                     (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),
                     255,
                     2)
        cv.imshow("debug", mask)
        cv.waitKey(0)
        #"""
        return bbox


    def project_sphere_around_kpt(self, kpt_3d, k_l, k_r):
        H_inv = np.hstack((self.H_inv, kpt_3d))
        H_inv_transp = np.transpose(H_inv)
        Q_ = H_inv_transp @ self.Q @ H_inv
        ellipse_1 = self.get_ellipse_param(self.P1, Q_, self.P1_transp)
        #ellipse_2 = self.get_ellipse_param(self.P2, Q_, self.P2_transp) # Does not work, I am not sure why
        """ Other way, translate the 3D point as it if the right camera was the main coordinate frame """
        kpt_3d[0,0] -= self.baseline
        P2 = self.P2.copy()
        P2[:, 3] = 0 # Set translation part of P2 to 0
        P2_transp = np.transpose(P2)
        H_inv = np.hstack((self.H_inv, kpt_3d))
        H_inv_transp = np.transpose(H_inv)
        Q_ = H_inv_transp @ self.Q @ H_inv
        ellipse_2 = self.get_ellipse_param(P2, Q_, P2_transp)
        #"""
        #print(ellipse_1)
        #print(ellipse_2)
        bbox1 = self.get_bbox_from_ellipse(ellipse_1)
        bbox2 = self.get_bbox_from_ellipse(ellipse_2)
        # Adjust bboxs so that the centre remains the same as the labelled one
        bbox1_half_w = int(bbox1[2] / 2)
        bbox1_half_h = int(bbox1[3] / 2)
        bbox1 = (k_l["u"] - bbox1_half_w,
                 k_l["v"] - bbox1_half_h,
                 bbox1_half_w * 2,
                 bbox1_half_h * 2)
        bbox2_half_w = int(bbox2[2] / 2)
        bbox2_half_h = int(bbox2[3] / 2)
        bbox2 = (k_r["u"] - bbox2_half_w,
                 k_r["v"] - bbox2_half_h,
                 bbox2_half_w * 2,
                 bbox2_half_h * 2)
        return bbox1, bbox2


    def project_3d_into_2d(self, kpt_3d, k_l, k_r):
        kpt_2d_l = self.P1 @ kpt_3d
        kpt_2d_r = self.P2 @ kpt_3d
        kpt_2d_l /= kpt_2d_l[2, 0]
        kpt_2d_r /= kpt_2d_r[2, 0]
        # Get bbox width by moving by `radius` in X
        kpt_3d[0,0] += self.radius
        kpt_2d_l2 = self.P1 @ kpt_3d
        kpt_2d_r2 = self.P2 @ kpt_3d
        kpt_2d_l2 /= kpt_2d_l2[2, 0]
        kpt_2d_r2 /= kpt_2d_r2[2, 0]
        bbox_size_l = int(round(kpt_2d_l2[0, 0] - kpt_2d_l[0, 0]))
        bbox_size_r = int(round(kpt_2d_r2[0, 0] - kpt_2d_r[0, 0]))
        # bbox tupple
        bbox1 = (k_l["u"] - bbox_size_l,
                 k_l["v"] - bbox_size_l,
                 bbox_size_l * 2,
                 bbox_size_l * 2)
        bbox2 = (k_r["u"] - bbox_size_r,
                 k_r["v"] - bbox_size_r,
                 bbox_size_r * 2,
                 bbox_size_r * 2)
        return bbox1, bbox2


    def start(self, ind_id):
        print("Get ground truth!")
        out_path = self.file_out.format(ind_id)
        # Loop through images (from 0 until the last frame)
        n_images = self.Images.get_n_im()
        data_kpt = {}
        for ind_im in range(n_images):
            # Get keypoint's 2D coordinates
            im_name = self.Images.get_im_pair_name(ind_im)
            self.Keypoints.update_ktp_pairs(im_name)
            k_l, k_r = self.Keypoints.get_kpts_given_ind_id(ind_id)
            if k_l is None or k_r is None \
               or not k_l["is_visible"] or not k_r["is_visible"]:
                data_kpt[ind_im] = None
                continue
            # Get keypoint's 3D point
            kpt_3d = self.get_kpt_3d_pt(k_l, k_r)
            # Project sphere into rectified image
            bboxs = self.project_sphere_around_kpt(kpt_3d, k_l, k_r)
            # Project 3D points into 2D to get bbox size
            #bboxs = self.project_3d_into_2d(kpt_3d, k_l, k_r)
            # Get bbox around mask
            data_kpt[ind_im] = bboxs, k_l["is_difficult"]
        print("Done!")
        utils.write_yaml_data(out_path, data_kpt)


class Draw:
    def __init__(self, config, v):
        self.ind_im = 0
        self.ind_id = 0
        self.is_zoom_on = False
        self.load_data_config(config, v)
        self.load_vis_config(config)
        self.mouse_u = 0
        self.mouse_v = 0
        self.is_mouse_on_im_l = False
        self.is_mouse_on_im_r = False
        self.initialize_im()
        self.range_start = -1
        self.range_end   = -1


    def load_data_config(self, config, v):
        c_data = config["data"]
        self.dir_data = c_data["dir"]
        # Images
        dir_l = os.path.join(self.dir_data, c_data["subdir_stereo_l"])
        dir_r = os.path.join(self.dir_data, c_data["subdir_stereo_r"])
        im_format = c_data["im_format"]
        self.Images = Images(dir_l, dir_r, im_format)
        # Keypoints
        dir_out_l = os.path.join(self.dir_data, c_data["subdir_output_l"])
        dir_out_r = os.path.join(self.dir_data, c_data["subdir_output_r"])
        self.Keypoints = Keypoints(dir_out_l, dir_out_r)
        # Interpolation
        self.Interpolation = Interpolation(self.Images, self.Keypoints)
        # Ground-truth
        gt_sph_rad_mm = c_data["gt_sphere_rad_mm"]
        file_out_gt = os.path.join(self.dir_data, c_data["file_output_gt"])
        self.GT = GT(v, self.Images, self.Keypoints, gt_sph_rad_mm, file_out_gt)


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
        self.kpt_color_s = c_kpt["color_s"]
        self.kpt_color_not_s = c_kpt["color_not_s"]
        self.kpt_s_thick_pxl = c_kpt["s_thick_pxl"]
        self.kpt_id_v_marg_pxl = c_kpt["id_v_marg_pxl"]
        c_zoom = c_vis["zoom"]
        self.zoom_color = c_zoom["color"]
        self.zoom_r_w_pxl_half = int(c_zoom["rect_w_pxl"] / 2.)
        self.zoom_r_h_pxl_half = int(c_zoom["rect_h_pxl"] / 2.)
        self.zoom_thick_pxl  = c_zoom["thick_pxl"]


    def initialize_im(self):
        self.n_im = self.Images.get_n_im()
        self.Images.im_update(self.ind_im)
        self.im_h, self.im_w = self.Images.get_resolution()
        self.zoom_kpt_l  = None
        self.zoom_kpt_r  = None
        self.update_im_with_keypoints(True)


    def copy_im_kpt_to_all(self):
        self.im_l_all = np.copy(self.im_l_kpt)
        self.im_r_all = np.copy(self.im_r_kpt)


    def im_draw_guide_line(self):
        self.copy_im_kpt_to_all() # Not to accumulate the guide lines
        line_thick = self.guide_t
        color = np.array(self.guide_c, dtype=np.uint8).tolist()
        v = self.mouse_v
        pt_l = (0, v)
        pt_r = (self.im_w, v)
        cv.line(self.im_l_all, pt_l, pt_r, color, line_thick)
        cv.line(self.im_r_all, pt_l, pt_r, color, line_thick)
        u = self.mouse_u
        pt_t = (u, 0)
        pt_b = (u, self.im_h)
        if self.is_mouse_on_im_l:
            cv.line(self.im_l_all, pt_t, pt_b, color, line_thick)
        elif self.is_mouse_on_im_r:
            cv.line(self.im_r_all, pt_t, pt_b, color, line_thick)


    def im_draw_kpt_cross(self, im, u, v, color, size_w, size_h, is_diff):
        # Draw outer square
        s_t = self.kpt_s_thick_pxl
        left_top = (u - size_w, v - size_h)
        right_bot = (u + size_w, v + size_h)
        cv.rectangle(im, left_top, right_bot, color, s_t)
        # Draw inner cross
        c_t = self.kpt_c_thick_pxl
        left_mid = (u - size_w, v)
        right_mid = (u + size_w, v)
        mid_top = (u, v - size_h)
        mid_bot = (u, v + size_h)
        if is_diff:
            pt1 = left_top
            pt2 = right_bot
            pt3 = (u - size_w, v + size_h) # left_bot
            pt4 = (u + size_w, v - size_h) # right_top
        else:
            pt1 = (u - size_w, v) # left_mid
            pt2 = (u + size_w, v) # right_mid
            pt3 = (u, v - size_h) # mid_top
            pt4 = (u, v + size_h) # mid_bot
        cv.line(im, pt1, pt2, color, c_t)
        cv.line(im, pt3, pt4, color, c_t)


    def im_draw_kpt_id(self, im, txt, u, v, color, size_w, size_h):
        left = u - size_w
        bot = v - size_h - self.kpt_id_v_marg_pxl
        font = cv.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale = self.get_text_scale_to_fit_height(txt, font, thickness)
        cv.putText(im, txt, (left, bot), font, font_scale, color, thickness)


    def im_draw_kpt_not_vis(self, im, color):
        s_t = self.kpt_s_thick_pxl
        cv.line(im, (0, 0), (self.im_w, self.im_h), color, s_t)
        cv.line(im, (self.im_w, 0), (0, self.im_h), color, s_t)


    def limit_u(self, u):
        if u < 0:
            return 0
        elif u > (self.im_w - 1):
            return (self.im_w - 1)
        return u


    def limit_v(self, v):
        if v < 0:
            return 0
        elif v > (self.im_h - 1):
            return (self.im_h - 1)
        return v


    def zoom_mode_get_rect(self, kpt):
        kpt_u = kpt["u"]
        kpt_v = kpt["v"]
        w_half = self.zoom_r_w_pxl_half
        h_half = self.zoom_r_h_pxl_half
        left = self.limit_u((kpt_u - w_half))
        top  = self.limit_v((kpt_v - h_half))
        right = self.limit_u((kpt_u + w_half))
        bot  = self.limit_v((kpt_v + h_half))
        return left, top, right, bot


    def im_draw_zoom_mode_rect(self, is_left):
        color = np.array(self.zoom_color, dtype=np.uint8).tolist()
        im = None
        kpt = None
        if is_left:
            kpt = self.zoom_kpt_l
            im = self.im_l_kpt
        else:
            kpt = self.zoom_kpt_r
            im = self.im_r_kpt
        if kpt is None:
            """
             Either the kpt is not visible,
              or the user went from first to the last image,
              or no labelled kpt was detected since the last `zoom_mode_reset()`
            """
            return
        left, top, right, bot = self.zoom_mode_get_rect(kpt)
        thick = self.zoom_thick_pxl
        # Trick for not showing the zoom rectangle on borders
        if left == 0:
            left -= thick
        if top == 0:
            top -= thick
        if right == (self.im_w - 1):
            right += thick
        if bot == (self.im_h - 1):
            bot += thick
        cv.rectangle(im, (left, top), (right, bot), color, thick)


    def zoom_mode_copy_kpt(self, is_left, kpt):
        if not kpt["is_visible"]:
            kpt = None
        if is_left:
            self.zoom_kpt_l = kpt
        else:
            self.zoom_kpt_r = kpt


    def im_draw_kpt_pair(self, ind_id, kpt, is_left, bbox=None):
        # Set color
        color = np.array(self.kpt_color_not_s, dtype=np.uint8).tolist()
        if ind_id == self.ind_id:
            self.n_kpt_selected += 1
            color = np.array(self.kpt_color_s, dtype=np.uint8).tolist()
            self.zoom_mode_copy_kpt(is_left, kpt)
        # Draw X if not visible and return
        is_visible = kpt["is_visible"]
        if not is_visible:
            if ind_id == self.ind_id: # Only if the ind_id is selected
                if self.is_zoom_on:
                    self.zoom_mode_reset()
                self.selected_id_not_visible = True
                if is_left:
                    self.im_draw_kpt_not_vis(self.im_l_kpt, color)
                else:
                    self.im_draw_kpt_not_vis(self.im_r_kpt, color)
            return
        # Draw keypoint (cross + id)
        txt = "{}".format(ind_id)
        kpt_u = kpt["u"]
        kpt_v = kpt["v"]
        is_interp = kpt["is_interp"]
        if is_interp:
            txt += "'" # If `is_interp` add a symbol
        if bbox is None:
            size_w = self.kpt_c_size_pxl
            size_h = size_w
        else:
            size_w = int(bbox[2] / 2)
            size_h = int(bbox[3] / 2)
        is_diff = kpt["is_difficult"]
        if is_left:
            self.im_draw_kpt_cross(self.im_l_kpt, kpt_u, kpt_v, color, size_w, size_h, is_diff)
            self.im_draw_kpt_id(self.im_l_kpt, txt, kpt_u, kpt_v, color, size_w, size_h)
        else:
            self.im_draw_kpt_cross(self.im_r_kpt, kpt_u, kpt_v, color, size_w, size_h, is_diff)
            self.im_draw_kpt_id(self.im_r_kpt, txt, kpt_u, kpt_v, color, size_w, size_h)


    def im_draw_all_kpts(self):
        kpts_l, kpts_r = self.Keypoints.get_kpts()
        self.n_kpt_selected = 0
        self.selected_id_not_visible = False
        no_pair_key = list(set(kpts_l.keys()).symmetric_difference(kpts_r.keys()))
        if no_pair_key:
            # There may be 1 kpt without pair (if being labelled)
            kpt_key = no_pair_key[0]
            no_pair_kpt = self.Keypoints.get_kpts_given_ind_id(kpt_key)
            if no_pair_kpt[1] is None:
                # It is on the left image
                self.im_draw_kpt_pair(kpt_key, no_pair_kpt[0], True)
            else:
                # It is on the right image
                kpt_r_val = kpts_r[kpt_key]
                self.im_draw_kpt_pair(kpt_key, no_pair_kpt[1], False)
        # Draw the paired keypoints
        for (kpt_l_key, kpt_l_val), (kpt_r_key, kpt_r_val) in zip(kpts_l.items(), kpts_r.items()):
            kpt_3d = self.GT.get_kpt_3d_pt(kpt_l_val, kpt_r_val)
            bboxs = self.GT.project_sphere_around_kpt(kpt_3d, kpt_l_val, kpt_r_val)
            self.im_draw_kpt_pair(kpt_l_key, kpt_l_val, True, bboxs[0])
            self.im_draw_kpt_pair(kpt_r_key, kpt_r_val, False, bboxs[1])
        # Draw zoom rectangle
        self.im_draw_zoom_mode_rect(True)
        self.im_draw_zoom_mode_rect(False)


    def zoom_mode_get_full_image_coords(self, u, v):
        rect_w = 2 * self.zoom_r_w_pxl_half
        is_mouse_on_right_crop = False
        if u < rect_w:
            left, top, right, bot = self.zoom_mode_get_rect(self.zoom_kpt_l)
        else:
            is_mouse_on_right_crop = True
            u -= rect_w
            left, top, right, bot = self.zoom_mode_get_rect(self.zoom_kpt_r)
        if u > (right - left) or v > (bot - top):
            return None, None
        u += left
        if is_mouse_on_right_crop:
            u += self.im_w
        v += top
        return u, v


    def update_mouse_position(self, u, v):
        if self.is_zoom_on:
            u, v = self.zoom_mode_get_full_image_coords(u, v)
        # Check if mouse is on left or right image
        self.is_mouse_on_im_l = False
        self.is_mouse_on_im_r = False
        if u is None or v is None:
            return
        if v < self.im_h:
            if u < self.im_w:
                self.is_mouse_on_im_l = True
            else:
                self.is_mouse_on_im_r = True
                u -= self.im_w
        # Force `v` if a point was already labelled
        if self.is_mouse_on_im_l:
            kpt_n_r = self.Keypoints.get_new_kpt_r()
            if kpt_n_r is not None:
                v = kpt_n_r["v"]
        elif self.is_mouse_on_im_r:
            kpt_n_l = self.Keypoints.get_new_kpt_l()
            if kpt_n_l is not None:
                v = kpt_n_l["v"]
        # Update position only if inside one of the images
        if self.is_mouse_on_im_l or\
           self.is_mouse_on_im_r:
            self.mouse_u = u
            self.mouse_v = v


    def mouse_move(self, u, v):
        self.update_mouse_position(u, v)
        self.im_draw_guide_line()


    def mouse_lclick(self):
        if self.selected_id_not_visible:
            return
        if self.is_mouse_on_im_l or self.is_mouse_on_im_r:
            if self.n_kpt_selected < 2: # If not already labelled
                # Save new keypoint
                self.Keypoints.new_kpt(self.is_mouse_on_im_l,
                                       self.ind_id,
                                       self.mouse_u,
                                       self.mouse_v)
                # Draw new keypoint as well
                self.update_im_with_keypoints(False)
            elif self.is_zoom_on:
                # Re-adjust label (only allowed in zoom mode)
                if self.is_mouse_on_im_l:
                    self.zoom_kpt_l["u"] = self.mouse_u
                else:
                    self.zoom_kpt_r["u"] = self.mouse_u
                self.zoom_kpt_l["v"] = self.mouse_v
                self.zoom_kpt_r["v"]  = self.mouse_v
                # Re-adjust left
                self.Keypoints.new_kpt(True,
                                       self.ind_id,
                                       self.zoom_kpt_l["u"],
                                       self.zoom_kpt_l["v"])
               # Re-adjust right
                self.Keypoints.new_kpt(False,
                                       self.ind_id,
                                       self.zoom_kpt_r["u"],
                                       self.zoom_kpt_r["v"])
                # Draw new keypoints as well
                self.update_im_with_keypoints(False)


    def get_text_scale_to_fit_height(self, txt, font, thickness):
        desired_height = self.bar_text_h_pxl
        _, text_h = cv.getTextSize(txt, font, 1.0, thickness)[0]
        scale = float(desired_height) / text_h
        return scale


    def get_text_width(self, txt, font, scale, thickness):
        text_w, _text_h = cv.getTextSize(txt, font, scale, thickness)[0]
        return text_w


    def add_status_text(self, bar):
        # Message
        txt = "Im: [{}/{}]".format(self.ind_im, self.n_im - 1)
        if self.range_start != -1:
            txt = "Im: [{} -> {}]".format(self.range_start, self.range_end)
        # Text specifications
        color = np.array(self.bar_text_c, dtype=np.uint8).tolist()
        font = cv.FONT_HERSHEY_DUPLEX
        thickness = 2
        font_scale = self.get_text_scale_to_fit_height(txt, font, thickness)
        # Centre text vertically
        left = self.bar_m_l_pxl
        bot = int((self.bar_h_pxl + self.bar_text_h_pxl) / 2.0)
        # Write text
        cv.putText(bar, txt, (left, bot), font, font_scale, color, thickness)
        left += self.get_text_width(txt, font, font_scale, thickness)
        txt = " Id: [{}]".format(self.ind_id)
        if self.n_kpt_selected > 0:
            color = np.array(self.kpt_color_s, dtype=np.uint8).tolist()
        cv.putText(bar, txt, (left, bot), font, font_scale, color, thickness)
        return bar


    def add_status_bar(self, draw):
        # Make black rectangle
        bar = np.zeros((self.bar_h_pxl, draw.shape[1], 3), dtype=draw.dtype)
        # Add text status to bar
        bar = self.add_status_text(bar)
        draw = np.concatenate((draw, bar), axis=0)
        return draw


    def update_im_with_keypoints(self, reload_kpt):
        im_l, im_r = self.Images.get_im_pair()
        self.im_l_kpt = np.copy(im_l)
        self.im_r_kpt = np.copy(im_r)
        if reload_kpt:
            self.load_kpt_data(self.ind_im)
        self.im_draw_all_kpts()
        self.copy_im_kpt_to_all()


    def load_kpt_data(self, ind_im):
        im_name = self.Images.get_im_pair_name(ind_im)
        self.Keypoints.update_ktp_pairs(im_name)


    def im_next(self):
        self.Keypoints.eliminate_unpaired_kpts()
        self.ind_im += 1
        if self.ind_im > (self.n_im - 1):
            self.ind_im = 0
            self.zoom_mode_reset()
        self.Images.im_update(self.ind_im)
        self.update_im_with_keypoints(True)


    def im_prev(self):
        self.Keypoints.eliminate_unpaired_kpts()
        self.ind_im -= 1
        if self.ind_im < 0:
            self.ind_im = (self.n_im - 1)
            self.zoom_mode_reset()
        self.Images.im_update(self.ind_im)
        self.update_im_with_keypoints(True)


    def id_next(self):
        self.Keypoints.eliminate_unpaired_kpts()
        self.ind_id += 1
        self.zoom_mode_reset()
        self.update_im_with_keypoints(False)


    def id_prev(self):
        self.Keypoints.eliminate_unpaired_kpts()
        self.ind_id -=1
        if self.ind_id < 0:
            self.ind_id = 0
        self.zoom_mode_reset()
        self.update_im_with_keypoints(False)


    def eliminate_selected_kpts(self):
        if self.selected_id_not_visible:
            return
        i_min, i_max = self.get_range_min_and_max()
        if i_min is not None and i_max is not None:
            for i in range(i_min, i_max + 1):
                self.load_kpt_data(i)
                self.Keypoints.eliminate_kpts(self.ind_id)
            self.update_im_with_keypoints(False)
            self.range_toggle()
        else:
            if self.n_kpt_selected > 0:
                self.Keypoints.eliminate_kpts(self.ind_id)
                self.update_im_with_keypoints(False)


    def interp_kpt_positions(self):
        if self.selected_id_not_visible:
            return
        self.Interpolation.start(self.ind_id, self.ind_im)
        """ Show the newly interpolated keypoints """
        self.update_im_with_keypoints(True)


    def get_range_min_and_max(self):
        i_min = None
        i_max = None
        if self.range_start != -1 and self.range_end != -1:
            i_min = min(self.range_start, self.range_end)
            i_max = max(self.range_start, self.range_end)
        return i_min, i_max


    def toggle_kpt_visibility(self):
        i_min, i_max = self.get_range_min_and_max()
        if i_min is not None and i_max is not None:
            for i in range(i_min, i_max + 1):
                self.load_kpt_data(i)
                self.Keypoints.toggle_is_visibile(self.ind_id)
            self.range_toggle()
        else:
            self.Keypoints.toggle_is_visibile(self.ind_id)
        self.update_im_with_keypoints(False)


    def toggle_kpt_difficult(self):
        i_min, i_max = self.get_range_min_and_max()
        if i_min is not None and i_max is not None:
            for i in range(i_min, i_max + 1):
                self.load_kpt_data(i)
                self.Keypoints.toggle_is_difficult(self.ind_id)
            self.range_toggle()
        else:
            self.Keypoints.toggle_is_difficult(self.ind_id)
        self.update_im_with_keypoints(False)


    def range_toggle(self):
        if self.range_start == -1:
            self.range_start = self.ind_im
            self.range_end   = self.ind_im
        else:
            self.range_start = -1
            self.range_end   = -1


    def range_update(self):
        if self.range_start != -1:
            self.range_end = self.ind_im


    def zoom_mode_toggle(self):
        if not self.is_zoom_on:
            if self.zoom_mode_check_start():
                self.is_zoom_on = True
        else:
            self.is_zoom_on = False


    def zoom_mode_check_start(self):
        if self.selected_id_not_visible:
            return False
        if self.zoom_kpt_l is None or \
           self.zoom_kpt_r is None:
            return False
        return True


    def zoom_mode_reset(self):
        self.is_zoom_on = False
        self.zoom_kpt_l = None
        self.zoom_kpt_r = None


    def zoom_mode_crop_im(self, im, kpt):
        left, top, right, bot = self.zoom_mode_get_rect(kpt)
        rect_w = 2 * self.zoom_r_w_pxl_half
        rect_h = 2 * self.zoom_r_h_pxl_half
        crop_im = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
        crop_im[:(bot - top),:(right - left)] = im[top:bot, left:right]
        return crop_im


    def get_draw(self):
        # Stack images together
        if self.is_zoom_on:
            im_l_crop = self.zoom_mode_crop_im(self.im_l_all, self.zoom_kpt_l)
            im_r_crop = self.zoom_mode_crop_im(self.im_r_all, self.zoom_kpt_r)
            draw = np.concatenate((im_l_crop, im_r_crop), axis=1)
        else:
            draw = np.concatenate((self.im_l_all, self.im_r_all), axis=1)
        # Add status bar in the bottom
        draw = self.add_status_bar(draw)
        return draw


    def save_gtruth(self):
        self.GT.start(self.ind_id)

class Interface:
    def __init__(self, config, v):
        self.load_keys_config(config)
        self.Draw = Draw(config, v)
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
        self.key_elimin  = c_keys["elimin"]
        self.key_interp  = c_keys["interp"]
        self.key_visibl  = c_keys["visible"]
        self.key_diffic  = c_keys["diffclt"]
        self.key_range   = c_keys["range"]
        self.key_zoom    = c_keys["zoom"]
        self.key_gtruth  = c_keys["gtruth"]


    def mouse_listener(self, event, x, y, flags, param):
        if (event == cv.EVENT_MOUSEMOVE):
            self.Draw.mouse_move(x, y)
        elif (event == cv.EVENT_LBUTTONUP):
            self.Draw.mouse_lclick()


    def create_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
        cv.setMouseCallback(self.window_name, self.mouse_listener)


    def check_key_pressed(self, key_pressed):
        if key_pressed == ord(self.key_im_next):
            self.Draw.im_next()
            self.Draw.range_update()
        elif key_pressed == ord(self.key_im_prev):
            self.Draw.im_prev()
            self.Draw.range_update()
        elif key_pressed == ord(self.key_id_next):
            self.Draw.id_next()
        elif key_pressed == ord(self.key_id_prev):
            self.Draw.id_prev()
        elif key_pressed == ord(self.key_elimin):
            self.Draw.eliminate_selected_kpts()
        elif key_pressed == ord(self.key_interp):
            self.Draw.interp_kpt_positions()
        elif key_pressed == ord(self.key_visibl):
            self.Draw.toggle_kpt_visibility()
        elif key_pressed == ord(self.key_diffic):
            self.Draw.toggle_kpt_difficult()
        elif key_pressed == ord(self.key_range):
            self.Draw.range_toggle()
        elif key_pressed == ord(self.key_zoom):
            self.Draw.zoom_mode_toggle()
        elif key_pressed == ord(self.key_gtruth):
            self.Draw.save_gtruth()


    def main_loop(self):
        """ Interface's main loop """
        key_pressed = None
        while key_pressed != ord(self.key_quit):
            draw = self.Draw.get_draw()
            cv.imshow(self.window_name, draw)
            key_pressed = cv.waitKey(1)
            self.check_key_pressed(key_pressed)


class Video:
    def __init__(self, calib_path, vid_path, vid_stack, is_to_rect, dir_l, dir_r, im_format):
        # Load calibration data
        self.load_calib_data(calib_path)
        self.stack_type = vid_stack
        self.get_im_size(vid_path)
        self.stereo_rectify()
        self.get_rectification_maps()
        # Get frames if needed
        self.is_to_rectify = is_to_rect
        self.get_frames_if_needed(dir_l, dir_r, vid_path, vid_stack, im_format)


    def get_im_size(self, vid_path):
        # Load first frame of video to get image size
        cap = cv.VideoCapture(vid_path)
        ret, frame = cap.read()
        if ret:
            self.im_h, self.im_w = frame.shape[:2]
            if self.stack_type == "vertical":
                self.im_h = int(self.im_h / 2)
            elif self.stack_type == "horizontal":
                self.im_w = int(self.im_w / 2)
            else:
                print("Error: unrecognized video type {}".format(self.stack_type))
        else:
            print("Error: failed to load video {}".format(vid_path))
            exit()
        cap.release()


    def load_calib_data(self, calib_path):
        fs = cv.FileStorage(calib_path, cv.FILE_STORAGE_READ)
        self.r = np.array(fs.getNode('R').mat(), dtype=np.float64)
        self.t = np.array(fs.getNode('T').mat()[0], dtype=np.float64)
        self.m1 = np.array(fs.getNode('M1').mat(), dtype=np.float64)
        self.d1 = np.array(fs.getNode('D1').mat()[0], dtype=np.float64)
        self.m2 = np.array(fs.getNode('M2').mat(), dtype=np.float64)
        self.d2 = np.array(fs.getNode('D2').mat()[0], dtype=np.float64)


    def stereo_rectify(self):
        self.R1, self.R2, self.P1, self.P2, self.Q, _roi1, _roi2 = \
            cv.stereoRectify(cameraMatrix1=self.m1,
                             distCoeffs1=self.d1,
                             cameraMatrix2=self.m2,
                             distCoeffs2=self.d2,
                             imageSize=(self.im_w, self.im_h),
                             R=self.r,
                             T=self.t,
                             flags=cv.CALIB_ZERO_DISPARITY,
                             alpha=0.0
                            )


    def get_rectification_maps(self):
        self.map1_x, self.map1_y = \
            cv.initUndistortRectifyMap(cameraMatrix=self.m1,
                                       distCoeffs=self.d1,
                                       R=self.R1,
                                       newCameraMatrix=self.P1,
                                       size=(self.im_w, self.im_h),
                                       m1type=cv.CV_32FC1
                                      )

        self.map2_x, self.map2_y = \
            cv.initUndistortRectifyMap(
                                       cameraMatrix=self.m2,
                                       distCoeffs=self.d2,
                                       R=self.R2,
                                       newCameraMatrix=self.P2,
                                       size=(self.im_w, self.im_h),
                                       m1type=cv.CV_32FC1
                                      )


    def split_frame(self, frame):
        if self.stack_type == "vertical":
            im1 = frame[:self.im_h, :]
            im2 = frame[self.im_h:, :]
        elif self.stack_type == "horizontal":
            im1 = frame[:, :self.im_w]
            im2 = frame[:, self.im_w:]
        else:
            print("Error: unrecognized stack type `{}`!".format(stack_type))
            exit()
        if self.is_to_rectify:
            im1 = cv.remap(im1, self.map1_x, self.map1_y, cv.INTER_LINEAR)
            im2 = cv.remap(im2, self.map2_x, self.map2_y, cv.INTER_LINEAR)
        return im1, im2


    def get_frames_if_needed(self, dir_l, dir_r, vid_path, vid_stack, im_format):
        if os.path.isdir(dir_l) and os.path.isdir(dir_r):
            return
        # Make output dirs
        os.mkdir(dir_l)
        os.mkdir(dir_r)
        # Go thourgh each frame
        print("Getting frames from video...")
        cap = cv.VideoCapture(vid_path)
        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            im1, im2 = self.split_frame(frame)
            im1_path = os.path.join(dir_l, "{:04}.png".format(frame_counter)) # TODO: hardcoded 4 padded zeros
            cv.imwrite(im1_path, im1)
            im2_path = os.path.join(dir_r, "{:04}.png".format(frame_counter))
            cv.imwrite(im2_path, im2)

            frame_counter += 1
        print("Finished!")
        cap.release()


def download_video_frames_and_rectify(config):
    # Download video into frames
    config_d = config['data']
    dir_data = config_d["dir"]
    calib_path = config_d["input_calib"]
    calib_path = os.path.join(dir_data, calib_path)
    vid_path = config_d["input_vid"]
    vid_path = os.path.join(dir_data, vid_path)
    vid_stack = config_d["vid_stack"]
    is_to_rect = config_d["is_to_rectify"]
    dir_l = config_d['subdir_stereo_l']
    dir_l = os.path.join(dir_data, dir_l)
    dir_r = config_d['subdir_stereo_r']
    dir_r = os.path.join(dir_data, dir_r)
    im_format = config_d['im_format']
    v = Video(calib_path, vid_path, vid_stack, is_to_rect, dir_l, dir_r, im_format)
    return v


def label_data(config):
    v = download_video_frames_and_rectify(config)
    inter = Interface(config, v)
    inter.main_loop()
