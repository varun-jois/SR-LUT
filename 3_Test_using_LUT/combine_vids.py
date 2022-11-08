import cv2 as cv
import numpy as np
from os.path import join

UTT_NAMES = ['id04119_1uH67UruKlE_00002', 'id07354_iUUpvrP-gzQ_00348', 'id08911_8QeBl-d07ik_00039']
FPATH = '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/vids/exp_2'
EXTS = ['_bic', '_lut', '_h264']

for i, utt_name in enumerate(UTT_NAMES, 1):
    id, vid, utt = utt_name.split('_')
    path_b = join(FPATH, f'{utt_name}_bic.mp4')
    path_l = join(FPATH, f'{utt_name}_lut.mp4')
    path_h = join(FPATH, f'{utt_name}_h264.mp4')

    # video file to read
    cap_b = cv.VideoCapture(path_b)
    cap_l = cv.VideoCapture(path_l)
    cap_h = cv.VideoCapture(path_h)

    # video files for the output
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # mp4v
    movie = cv.VideoWriter(join(FPATH, f'{utt}_blh.mp4'), fourcc, 25, (672, 224))
    while True:
        # Capture frame-by-frame
        ret_b, f_b = cap_b.read()
        ret_l, f_l = cap_l.read()
        ret_h, f_h = cap_h.read()
        if ret_b and ret_l and ret_h:
            # concatenate frames
            frame = np.concatenate((f_b, f_l, f_h), axis=1)
            movie.write(frame)
        else:
            break
    # When everything done, release the capture
    cap_b.release()
    cap_l.release()
    cap_h.release()
    movie.release()
    cv.destroyAllWindows()
    print(f'{i} of {len(UTT_NAMES)} done')
