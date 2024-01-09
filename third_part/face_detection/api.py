from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


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

class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

ROOT = os.path.dirname(os.path.abspath(__file__))

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))

        return results
    def get_detections_for_batch_withcoords(self, images, coords):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for d, coord in zip(detected_faces,coords):
            assert len(d) > 0, "[get_detections_for_batch_withcoords]:face should be contain in this frame due to the face preprocesser"
            
            max_iou = -1
            cur_d = None
            for temp_d in d:
                temp_d = np.clip(temp_d, 0, None)
                x1, y1, x2, y2 = map(int, temp_d[:-1])
                cur_coord = (x1, y1, x2, y2)
                cur_iou = get_iou(cur_coord, coord)
                if cur_iou > max_iou:
                    max_iou = cur_iou
                    cur_d = cur_coord
            results.append(cur_d)
                
        return results