import importlib
import os

import numpy as np
import cv2 as cv
from glob import glob
from PIL import Image
#
# import sys
# sys.path.insert(1, '1_Train_deep_model')
# from utils import PSNR, _rgb2ycbcr
#
#
# fpath = '/home/varun/PhD/datasets/VoxCeleb2/vox2_test_mp4/mp4/id00017/01dfn2spqyE/00001.mp4'
# out_path = '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/test_me'
#
# # read the video and iterate over the frames
# cap = cv.VideoCapture(fpath)
# fc = 1
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if ret:
#         fc += 1
#         if fc % 25 == 0:
#             cv.imwrite(f'{out_path}/HR/{fc}_GT.png', frame)
#             frame = cv.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv.INTER_CUBIC)
#             cv.imwrite(f'{out_path}/LR_x4/{fc}.png', frame)
#     else:
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()
#
# # bicubic upsampling
# files_lr = glob(out_path + '/LR_x{}/*.png'.format(4))
# files_lr.sort()
# files_gt = glob(out_path + '/HR/*.png')
# files_gt.sort()
# psnrs = []
#
# for ti, fn in enumerate(files_gt):
#     # Load LR image
#     img_lr = np.array(Image.open(files_lr[ti]))#.astype(np.float32)
#     h, w, c = img_lr.shape
#
#     # Load GT image
#     img_gt = np.array(Image.open(files_gt[ti]))
#
#     name = os.path.basename(files_lr[ti])
#
#     # bicubic upsampling
#     img_out = cv.resize(img_lr, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
#     cv.imwrite(f'/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/output_bicubic/{name}', img_out[...,::-1])
#
#     CROP_S = 4
#     psnr = PSNR(_rgb2ycbcr(img_gt)[:, :, 0], _rgb2ycbcr(img_out)[:, :, 0], CROP_S)
#     psnrs.append(psnr)
#
# print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))
#
# # x = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
# # y = cv.resize(x, (0, 0), fx=1/4, fy=1/4, interpolation=cv.INTER_AREA)
#
# paths = glob('/home/varun/PhD/datasets/VoxCeleb2/vox2_test_mp4/mp4/*')
# test = np.random.choice(paths, 3, replace=False)
# [v for p in test for v in glob(f'{p}/*')]
#
# test_ids = ['id08911', 'id04119', 'id07354']
#

VID_PATH = '/home/varun/fvc/datasets/vox2_test_mp4_lr_x4'
lrpath = '1_Train_deep_model/train/DIV2K_train_LR_bicubic/X4'
fs = ['id04295_JPMZAOGGHh8_00073', 'id04295_pF1eg2ltAIk_00212', 'id04295_rBoL5RHpTx0_00217']
for f in fs:
    id, vid, utt = f.split('_')
    uttpath = f'{VID_PATH}/{id}/{vid}/{utt}.mp4'
    # go through the video
    cap = cv.VideoCapture(uttpath)
    fc = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if fc == 10:  # only taking one frame, the tenth frame
                name = f'{id}_{vid}_{utt}_{fc}'
                # write the LR image
                cv.imwrite(f'{lrpath}/{name}.png', frame)
                break
            fc += 1
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()



