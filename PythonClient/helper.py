# coding: utf-8
import math
import cv2
import numpy as np
import argparse
import os
import sys
from scipy.spatial import distance as dist


def start_up_init():
    parser = argparse.ArgumentParser(description='ArcFace Online Test')

    # =================== General ARGS ====================
    parser.add_argument('--max_face_number',
                        type=int,
                        help='同时检测的最大人脸数量',
                        default=8)
    parser.add_argument('--max_frame_rate', type=int, help='最大FPS', default=25)
    parser.add_argument('--image_size',
                        default='112,112',
                        help='输入特征提取网络的图片大小')
    parser.add_argument('--retina_model',
                        default='./model/M25',
                        help='人脸检测网络预训练模型路径')
    parser.add_argument('--gpu', default=0, type=int, help='GPU设备ID，-1代表使用CPU')
    parser.add_argument('--threshold',
                        default=0.7,
                        type=float,
                        help='RetinaNet的人脸检测阈值')
    parser.add_argument('--scales',
                        type=float,
                        nargs='+',
                        help='RetinaNet的图像缩放系数',
                        default=[0.3])

    return parser.parse_args()


def encode_image(image, quality=80):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    return cv2.imencode('.jpg', image, encode_param)[1].tostring()


def draw_points(image, poi, margin=5, color=[255, 255, 0]):
    for index in range(5):
        image[poi[index, 1] - margin:poi[index, 1] + margin, poi[index, 0] -
              margin:poi[index, 0] + margin] = color


K = [
    6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0,
    6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0
]
D = [
    7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0,
    -1.3073460323689292e+000
]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0], [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0], [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0], [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    image_pts = np.float32([
        shape[17], shape[21], shape[22], shape[26], shape[36], shape[39],
        shape[42], shape[45], shape[31], shape[35], shape[48], shape[54],
        shape[57], shape[8]
    ])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts,
                                                    cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec,
                                        translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
