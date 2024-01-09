import os
import cv2
import time
import glob
import argparse
import scipy
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import cycle
from torch.multiprocessing import Pool, Process, set_start_method


"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from 
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html
requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from: 
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import numpy as np
from PIL import Image
import dlib

# rect1/rect2 is [x_top_left,y_top_left,x_bottom_right,y_bottom_right]
def get_iou(rect1, rec1):
    x1, x2, y1, y2 = rect1
    xx1, xx2, yy1, yy2 = rec1
    # determine the coordinates of the intersection rectangle
    x_left = max(x1, xx1)
    y_top = max(y1, yy1)
    x_right = min(x2, xx2)
    y_bottom = min(y2, yy2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # compute the area of both rectangles
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (xx2 - xx1) * (yy2 - yy1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

class Croper:
    def __init__(self, path_of_lm):
        # download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor(path_of_lm)

    def get_landmark(self, img_np):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()
        dets = detector(img_np, 1)
        if len(dets) == 0:
            return None
        d = dets[0]
        # Get the landmarks/parts for the face in box d.
        shape = self.predictor(img_np, d)
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm
    def get_uni_landmark_withcoord(self, img_np, coord):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        detector = dlib.get_frontal_face_detector()
        dets = detector(img_np, 1)
        assert len(dets) > 0, "[get_uni_landmark_withcoord]:face should be contain in this frame due to the face preprocesser"
        
        max_iou = -1
        cur_d = None
        for i in range(len(dets)):
            cur_xyxy = [dets[i].tl_corner().x, dets[i].tl_corner().y, dets[i].br_corner().x, dets[i].br_corner().y]
            cur_iou = get_iou(cur_xyxy, coord)
            if cur_iou > max_iou:
                max_iou = cur_iou
                cur_d = dets[i]
        d = cur_d
        # Get the landmarks/parts for the face in box d.
        shape = self.predictor(img_np, d)
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    def align_face(self, img, lm, output_size=1024):
        """
        :param filepath: str
        :return: PIL Image
        """
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]  
        x /= np.hypot(*x)  
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)   
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])   
        qsize = np.hypot(*x) * 2   

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            quad -= crop[0:2]

        # Transform.
        quad = (quad + 0.5).flatten()
        lx = max(min(quad[0], quad[2]), 0)
        ly = max(min(quad[1], quad[7]), 0)
        rx = min(max(quad[4], quad[6]), img.size[0])
        ry = min(max(quad[3], quad[5]), img.size[0])

        # Save aligned image.
        return crop, [lx, ly, rx, ry]
    
    def crop(self, img_np_list, xsize=512):    # first frame for all video
        idx = 0
        while idx < len(img_np_list)//2 :   # TODO 
            img_np = img_np_list[idx]
            lm = self.get_landmark(img_np)
            if lm is not None:  
                break   # can detect face
            idx += 1
        if lm is None:
            return None
        
        crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
        clx, cly, crx, cry = crop
        lx, ly, rx, ry = quad
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        for _i in range(len(img_np_list)):
            _inp = img_np_list[_i]
            _inp = _inp[cly:cry, clx:crx]
            _inp = _inp[ly:ry, lx:rx]
            img_np_list[_i] = _inp
        return img_np_list, crop, quad
    
    def crop_eachframe(self, img_np_list, coords_list, xsize=512):    # first frame for all video
        
        crop_list = []
        quad_list = []
        for _i in tqdm(range(len(img_np_list)),desc="crop_eachframe"):
            img_np = img_np_list[_i]
            coord = coords_list[_i]
            lm = self.get_uni_landmark_withcoord(img_np,coord)
            assert lm is not None, "[crop_eachframe],face should be contain in this frame due to the face preprocesser"
        
            crop, quad = self.align_face(img=Image.fromarray(img_np), lm=lm, output_size=xsize)
            clx, cly, crx, cry = crop
            lx, ly, rx, ry = quad
            lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
            
            _inp = img_np_list[_i]
            _inp = _inp[cly:cry, clx:crx]
            _inp = _inp[ly:ry, lx:rx]
            img_np_list[_i] = _inp
            
            crop_list.append(crop)
            quad_list.append(quad)
            
        return img_np_list, crop_list, quad_list


