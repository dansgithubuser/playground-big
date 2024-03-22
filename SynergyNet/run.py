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

#===== args =====#
parser = argparse.ArgumentParser()
parser.add_argument('input_path')
args = parser.parse_args()

#===== main =====#
model = SynergyNet()
cap = cv2.VideoCapture(args.input_path)
while True:
    ret, im = cap.read()
    if not ret: break
    lmk3d, mesh, pose = model.get_all_outputs(im)
    print('=====')
    print(lmk3d)
    print(mesh)
    print(pose)
