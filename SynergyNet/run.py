#!/usr/bin/env python3

#===== early =====#
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent / 'SynergyNet'))

#===== imports =====#
#----- in-repo -----#
from synergy3DMM import SynergyNet

#----- 3rd party -----#
import cv2

#----- standard -----#
import argparse
from math import cos, pi, sin, sqrt

#===== args =====#
parser = argparse.ArgumentParser()
parser.add_argument('input_path')
args = parser.parse_args()

#===== helpers =====#
def draw_axis(im, yaw, pitch, roll, tdx=None, tdy=None, size=100, pts68=None):
    pitch =  pitch * pi / 180
    yaw   = -yaw   * pi / 180
    roll  =  roll  * pi / 180

    if tdx != None and tdy != None:
        pass
    else:
        height, width = im.shape[:2]
        tdx = width / 2
        tdy = height / 2
    if pts68:
        tdx = pts68[0,30]
        tdy = pts68[1,30]
        minx, maxx = min(pts68[0, :]), max(pts68[0, :])
        miny, maxy = min(pts68[1, :]), max(pts68[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))
        size = llength * 0.5

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(im, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(im, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.line(im, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)

    return im

#===== main =====#
model = SynergyNet()
cap = cv2.VideoCapture(args.input_path)
while True:
    ret, im = cap.read()
    if not ret:
        cv2.waitKey()
        break
    lmk3ds, meshes, poses = model.get_all_outputs(im)
    for angles, translations in poses:
        im = draw_axis(im, angles[0], angles[1], angles[2], translations[0], translations[1])
    cv2.imshow('SynergyNet', im)
    cv2.waitKey(1)
