# OpenVtuber-虚拟爱抖露共享计划

## 运行方法（Easy Start）

* `node ./NodeServer/server.js`
* `python3.7 ./PythonClient/vtuber_usb_camera.py --gpu -1`


## 人脸检测 （Face Detection）
* [MTCNN (Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks)](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)
* [MTCNN (mxnet version)](https://github.com/deepinsight/insightface)


## 头部姿态估计（Head Pose Estimation）
* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)


## 特征点检测（Facial Landmarks Tracking）
* [Download Landmarks Model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
* Using dlib for facial features tracking
* Algorithm from [CVPR 2014](http://www.csc.kth.se/~vahidk/papers/KazemiCVPR14.pdf)
* Training set is based on i-bug 300-W datasets. It's annotation is shown below:<br><br>
![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)

## 表情识别（Emotion Recognition）

- [face_classification](https://github.com/oarriaga/face_classification)
- IMDB gender classification test accuracy: 96%.
- fer2013 emotion classification test accuracy: 66%.

## Live2D

- [插件版本](https://github.com/EYHN/hexo-helper-live2d)
- [打包版本](https://github.com/galnetwen/Live2D)

## 推流框架

- [MJPEG Framework](https://github.com/1996scarlet/MJPEG_Framework)

## FAQ

* Why not RetinaFace ?

    | Methods | LFW | CFP-FP | AgeDB-30
    | --------|-----|--------|---------
    | MTCNN+ArcFace | 99.83 | 98.37 | 98.15
    | RetinaFace+ArcFace | 99.86 | 99.49 | 98.60


## Citation

```
@article{7553523,
    author={K. Zhang and Z. Zhang and Z. Li and Y. Qiao}, 
    journal={IEEE Signal Processing Letters}, 
    title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks}, 
    year={2016}, 
    volume={23}, 
    number={10}, 
    pages={1499-1503}, 
    keywords={Benchmark testing;Computer architecture;Convolution;Detectors;Face;Face detection;Training;Cascaded convolutional neural network (CNN);face alignment;face detection}, 
    doi={10.1109/LSP.2016.2603342}, 
    ISSN={1070-9908}, 
    month={Oct}
}
```