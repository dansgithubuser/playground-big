#!/usr/bin/env python3

import cv2
from sixdrepnet import SixDRepNet

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_path')
args = parser.parse_args()

cap = cv2.VideoCapture(args.input_path)
model = SixDRepNet(gpu_id=-1)

while True:
    ret, im = cap.read()
    if not ret: break
    pitch, yaw, roll = model.predict(im)
    model.draw_axis(im, yaw, pitch, roll)
    cv2.imshow("6D rep net", im)
    cv2.waitKey(1)
cv2.waitKey()
