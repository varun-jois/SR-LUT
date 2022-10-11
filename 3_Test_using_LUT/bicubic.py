import os

import numpy as np
import cv2 as cv
from glob import glob
from PIL import Image

import sys
sys.path.insert(1, '../1_Train_deep_model')
from utils import PSNR, _rgb2ycbcr

out_path = '/home/varun/fvc/SR-LUT/3_Test_using_LUT/test'

files_lr = glob(out_path + '/LR_x{}/*.png'.format(4))
files_lr.sort()
files_gt = glob(out_path + '/HR/*.png')
files_gt.sort()
psnrs = []

for ti, fn in enumerate(files_gt):
    # Load LR image
    img_lr = np.array(Image.open(files_lr[ti]))#.astype(np.float32)
    h, w, c = img_lr.shape

    # Load GT image
    img_gt = np.array(Image.open(files_gt[ti]))

    name = os.path.basename(files_lr[ti])

    # bicubic upsampling
    img_out = cv.resize(img_lr, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
    cv.imwrite(f'/home/varun/fvc/SR-LUT/3_Test_using_LUT/output_bicubic/{name}', img_out[...,::-1])

    CROP_S = 4
    psnr = PSNR(_rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0], CROP_S)
    psnrs.append(psnr)

print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))
