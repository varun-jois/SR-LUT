from os.path import join
import cv2 as cv
from lut_sr import sr

IDS_PATH = '/home/varun/fvc/datasets/vox2_test_mp4_lr_x4'
LUT_PATH = '/home/varun/fvc/SR-LUT/2_Transfer_to_LUT/LUTs/Model_S_id04119.npy'
# IDS_PATH = '/home/varun/PhD/datasets/VoxCeleb2/vox2_test_mp4/mp4'  # '/home/varun/fvc/datasets/vox2_test_mp4'
# LUT_PATH = '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/Model_S_faces.npy'  #  '/home/varun/PhD/Face Video Compression/SR-LUT/3_Test_using_LUT/Model_S_faces.npy'
# UTT_NAMES = ['id04119_1uH67UruKlE_00002_10', 'id07354_iUUpvrP-gzQ_00348_10', 'id08911_8QeBl-d07ik_00039_10']
UTT_NAMES = ['id04119_1uH67UruKlE_00002_10']
OUT_PATH = 'vids'

for i, utt_name in enumerate(UTT_NAMES, 1):
    id, vid, utt, _ = utt_name.split('_')
    upath = join(IDS_PATH, id, vid, f'{utt}.mp4')

    # video file to read
    cap = cv.VideoCapture(upath)

    # video files for the output
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # mp4v
    # movie_bic = cv.VideoWriter(join(OUT_PATH, f'{id}_{vid}_{utt}_bic.mp4'), fourcc, 25, (224, 224))
    movie_lut = cv.VideoWriter(join(OUT_PATH, f'{id}_{vid}_{utt}_lut.mp4'), fourcc, 25, (224, 224))
    while True:
        # Capture frame-by-frame
        ret, lr = cap.read()
        if ret:
            # get the LR image
            # lr = cv.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv.INTER_CUBIC)
            # movie_lr.write(lr)
            # get bicubic upsampling
            # sr_bic = cv.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
            # # sr_bic = cv.resize(sr_bic, (0, 0), fx=4, fy=4, interpolation=cv.INTER_CUBIC)
            # movie_bic.write(sr_bic)
            # get lut output
            sr_lut = sr(lr[..., ::-1], LUT_PATH)
            # sr_lut = sr(sr_lut[..., ::-1], LUT_PATH)
            movie_lut.write(sr_lut)
        else:
            break
    # When everything done, release the capture
    cap.release()
    # movie_bic.release()
    movie_lut.release()
    cv.destroyAllWindows()

    print(f'{i} of {len(UTT_NAMES)} done')
