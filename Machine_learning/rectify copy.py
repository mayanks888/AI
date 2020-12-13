#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Float32MultiArray
from cv_bridge import CvBridge
import cv_bridge
# General libs
import numpy as np
import os
import time
import cv2
import torch
import shutil
import argparse
from sys import platform
from .models import *
from .utils import *
from .torch_utils import *
from std_msgs.msg import String


class Rectifier(Node):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--cfg', type=str, default='/module/src/rectifier/rectifier/cfg/yolov3.cfg', help='*.cfg path')
        parser.add_argument('--names', type=str,
                            default='data/coco.names', help='*.names path')
        parser.add_argument('--weights', type=str,
                            default='/module/src/rectifier/rectifier/weights/yolov3.pt', help='weights path')
        parser.add_argument('--source', type=str,
                            default='data/samples', help='source')
        parser.add_argument('--output', type=str, default='output',
                            help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=416,
                            help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.3, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.6, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v',
                            help='output video codec (verify ffmpeg support)')
        parser.add_argument('--half', action='store_true',
                            help='half precision FP16 inference')
        parser.add_argument('--device', default='0',
                            help='device id (i.e. 0 or 0,1) or cpu')
        parser.add_argument('--view-img', action='store_true',
                            help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--classes', nargs='+',
                            type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true',
                            help='class-agnostic NMS')
        self.opt = parser.parse_args()
        # self.opt = args_
        print(self.opt)
        # (320, 192) or (416, 256) or (608, 352) for (height, width)
        img_size = (320, 192) if ONNX_EXPORT else self.opt.img_size
        out, source, weights, self.half, view_img, save_txt = self.opt.output, self.opt.source, self.opt.weights, self.opt.half, self.opt.view_img, self.opt.save_txt

        self.device = select_device(
            device='cpu' if ONNX_EXPORT else self.opt.device)
        self.model = Darknet(self.opt.cfg, img_size)
        self.model.load_state_dict(torch.load(
            weights, map_location=self.device)['model'])
        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(self.device).eval()
        # Eval mode
        self.model.to(self.device).eval()
        self.tl_bbox = Float32MultiArray()
        self.arry_size = 6
        super().__init__('talker')
        self._pub = self.create_publisher(
            Float32MultiArray, '/tl_bbox_info', 10)
        self._cv_bridge = CvBridge()
        super().__init__('rectifier')
        self._sub = self.create_subscription(
            Image, '/snowball/perception/traffic_light/cropped_roi', self.img_callback, 10)
        print("ready to process rectifier----------------------------------------------------------")

    # Camera image callback
    def img_callback(self, image_msg):
        t = time.time()
        bridge = cv_bridge.CvBridge()
        cv_img = bridge.imgmsg_to_cv2(image_msg, 'passthrough')
        print("image shape is ", cv_img.shape)
        image_np = cv_img
        img, im0s = load_image_direct(image_np, img_size=416)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred = self.model(img)[0].float() if self.half else self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                   classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        tmp = [-1.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        if pred[0] is not None:
            print('prediction before', pred[0])
            # to select only traffic light
            keep = torch.where(pred[0][:, 5] == 9)[0]
            pred[0] = pred[0][keep]
            print('prediction after', pred[0])
            # Process detections
            for loop, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    tmp = -np.ones(self.arry_size * int(len(det)) + 1)
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0s.shape).round()
                    for i, boxes in enumerate(det):
                        c1, c2 = (int(boxes[0]), int(
                            boxes[1])), (int(boxes[2]), int(boxes[3]))
                        cv2.rectangle(image_np, c1, c2, color=(
                            255, 0, 0), thickness=2)
                        tmp[self.arry_size * i + 1] = float(boxes[0])
                        tmp[self.arry_size * i + 2] = float(boxes[1])
                        tmp[self.arry_size * i + 3] = float(boxes[2]-boxes[0])
                        tmp[self.arry_size * i + 4] = float(boxes[3]-boxes[1])
                        tmp[self.arry_size * i + 5] = float(boxes[4])
                        tmp[self.arry_size * i + 6] = float(1)
                        # tmp = tmp.tolist()
                    tmp = tmp.tolist()
                    cv2.imshow('AI rectifier', image_np)
                    ch = cv2.waitKey(1)
        else:
            tmp = [-1.0, 10.0, 10.0, 10.0, 10.0, 10.0]

        self.tl_bbox.data = tmp
        thorth = time.time()
        time_cost = round((thorth-t)*1000)
        print("Inference Time", time_cost)
        print("publishing bbox:- ", self.tl_bbox.data)
        self._pub.publish(self.tl_bbox)


def main(args=None):
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--cfg', type=str, default='/module/src/rectifier/rectifier/cfg/yolov3.cfg', help='*.cfg path')
    # parser.add_argument('--names', type=str,
    #                     default='data/coco.names', help='*.names path')
    # parser.add_argument('--weights', type=str,
    #                     default='/module/src/rectifier/rectifier/weights/yolov3.pt', help='weights path')
    # parser.add_argument('--source', type=str,
    #                     default='data/samples', help='source')
    # parser.add_argument('--output', type=str, default='output',
    #                     help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=416,
    #                     help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float,
    #                     default=0.3, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float,
    #                     default=0.6, help='IOU threshold for NMS')
    # parser.add_argument('--fourcc', type=str, default='mp4v',
    #                     help='output video codec (verify ffmpeg support)')
    # parser.add_argument('--half', action='store_true',
    #                     help='half precision FP16 inference')
    # parser.add_argument('--device', default='0',
    #                     help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--view-img', action='store_true',
    #                     help='display results')
    # parser.add_argument('--save-txt', action='store_true',
    #                     help='save results to *.txt')
    # parser.add_argument('--classes', nargs='+',
    #                     type=int, help='filter by class')
    # parser.add_argument('--agnostic-nms', action='store_true',
    #                     help='class-agnostic NMS')
    # opt = parser.parse_args()
    # # print(opt)
    rclpy.init(args=args)
    # with torch.no_grad():
    node = Rectifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
