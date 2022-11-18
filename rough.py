import importlib
from os.path import join

import numpy as np
import cv2 as cv
from glob import glob
from PIL import Image
import os
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

# import sys
# sys.path.insert(1, '3_Test_using_LUT')
# from lut_sr import sr
#
# IDS_PATH = '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/vids/exp_2'
# LUT_PATH = '/home/varun/PhD/Face Video Compression/SR-LUT/2_Transfer_to_LUT/LUTs/Model_S_faces_h264.npy'
# # IDS_PATH = '/home/varun/PhD/datasets/VoxCeleb2/vox2_test_mp4/mp4'  # '/home/varun/fvc/datasets/vox2_test_mp4'
# # LUT_PATH = '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/Model_S_faces.npy'  #  '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/Model_S_faces.npy'
# UTT_NAMES = ['id04119_1uH67UruKlE_00002', 'id07354_iUUpvrP-gzQ_00348', 'id08911_8QeBl-d07ik_00039']
# OUT_PATH = IDS_PATH
#
# for i, utt_name in enumerate(UTT_NAMES, 1):
#     upath = join(IDS_PATH, f'{utt_name}_lut.mp4')
#
#     # video file to read
#     cap = cv.VideoCapture(upath)
#     f = 1
#
#     # video files for the output
#     fourcc = cv.VideoWriter_fourcc(*'mp4v')  # mp4v
#     # movie_bic = cv.VideoWriter(join(OUT_PATH, f'{id}_{vid}_{utt}_bic.mp4'), fourcc, 25, (224, 224))
#     movie_lut = cv.VideoWriter(join(OUT_PATH, f'{utt_name}_lut16.mp4'), fourcc, 25, (896, 896))
#     while True:
#         # Capture frame-by-frame
#         ret, lr = cap.read()
#         if ret:
#             # get the LR image
#             # lr = cv.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv.INTER_CUBIC)
#             # movie_lr.write(lr)
#             # get bicubic upsampling
#             # sr_bic = cv.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
#             # # sr_bic = cv.resize(sr_bic, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
#             # movie_bic.write(sr_bic)
#             # get lut output
#             sr_lut = sr(lr[..., ::-1], LUT_PATH)
#             # sr_lut = sr(sr_lut[..., ::-1], LUT_PATH)
#             movie_lut.write(sr_lut)
#             print(f'{f} frames done')
#             f += 1
#         else:
#             break
#     # When everything done, release the capture
#     cap.release()
#     # movie_bic.release()
#     movie_lut.release()
#     cv.destroyAllWindows()
#
#     print(f'{i} of {len(UTT_NAMES)} done')


lrpath = '1_Train_deep_model/overfit/colors/DIV2K_train_LR_bicubic/X4'
hrpath = '1_Train_deep_model/overfit/colors/DIV2K_train_HR'
if not os.path.exists(lrpath):
    os.makedirs(lrpath)
if not os.path.exists(hrpath):
    os.makedirs(hrpath)

a = np.zeros((48, 48, 3), dtype=np.uint8)
a[..., 0] = 255
a[..., 1] = 255
Image.fromarray(a[:24, :24, :], 'RGB').save('timg.png')

for i in range(120):
    a = np.zeros((224, 224, 3), dtype=np.uint8)
    a[..., 0] = np.random.randint(256)
    a[..., 1] = np.random.randint(256)
    a[..., 2] = np.random.randint(256)
    Image.fromarray(a, 'RGB').save(f'{hrpath}/{i}_GT.png')
    Image.fromarray(a[:56, :56, :], 'RGB').save(f'{lrpath}/{i}.png')
