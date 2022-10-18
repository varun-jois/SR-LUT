from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
import cv2 as cv

IDS_PATH = '/home/varun/fvc/datasets/vox2_test_mp4'
UTT_NAMES = ['id04119_1uH67UruKlE_00002_10', 'id07354_iUUpvrP-gzQ_00348_10', 'id08911_8QeBl-d07ik_00039_10']

for utt_name in UTT_NAMES:
    id, vid, utt, _ = utt_name.split('_')
    upath = join(IDS_PATH, id, vid, f'{utt}.mp4')

    # video file to read
    cap = cv.VideoCapture(upath)

    # video files for the output
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    movie_bic = cv.VideoWriter(f'{id}_{vid}_{utt}_bic.mp4', fourcc, 25, (224, 224))
    movie_lut = cv.VideoWriter(f'{id}_{vid}_{utt}_lut.mp4', fourcc, 25, (224, 224))
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # get the LR image
            lr = cv.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv.INTER_CUBIC)
            # get bicubic upsampling
            bic_sr = cv.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
            movie_bic.write(bic_sr)
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()




