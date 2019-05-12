import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
from mtcnn_detector import MtcnnDetector

import face_preprocess

class DetectorModel:
    def __init__(self, args):
        ctx = mx.cpu() if args.gpu == -1 else mx.gpu(args.gpu)
        mtcnn_path = os.path.join(os.path.dirname(__file__), './zoo/mtcnn-model')

        self.max_face_number = args.max_face_number
        self.counter = 0
        self.detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1,
                                      minsize=args.mtcnn_minsize, factor=args.mtcnn_factor,
                                      accurate_landmark=True, threshold=args.mtcnn_threshold)

    def save_image(self, image):
        cv2.imwrite('./Temp/{}-{}.jpg'.format(time.time(), self.counter), image)
        self.counter += 1

    def get_all_boxes(self, face_img, save_img=False):
        face_num = self.max_face_number
        ret = self.detector.detect_face(face_img, det_type=0)
        if ret is None:
            return []

        bbox, points = ret

        sorted_index = bbox[:, 0].argsort()
        bbox = bbox[sorted_index]
        points = points[sorted_index]

        aligned = []
        for index in range(0, len(bbox[:face_num])):
            item_bbox = bbox[index, 0:4]
            item_points = points[index, :].reshape((2, 5)).T
            nimg = face_preprocess.preprocess(
                face_img, item_bbox, item_points, image_size='112,112')

            if save_img:
                save_image(nimg)

            aligned.append(nimg)

        return zip(aligned, bbox)

    def get_boxes_and_landmarks(self, face_img, save_img=False):
        face_num = self.max_face_number
        ret = self.detector.detect_face(face_img, det_type=0)
        if ret is None:
            return []

        bbox, points = ret

        sorted_index = bbox[:, 0].argsort()
        bbox = bbox[sorted_index]
        points = points[sorted_index]

        aligned = []
        pois = []
        for index in range(0, len(bbox[:face_num])):
            item_bbox = bbox[index, 0:4]
            item_points = points[index, :].reshape((2, 5)).T
            nimg = face_preprocess.preprocess(
                face_img, item_bbox, item_points, image_size='112,112')

            if save_img:
                save_image(nimg)

            aligned.append(nimg)
            pois.append(item_points)

        return zip(aligned, bbox, pois)
