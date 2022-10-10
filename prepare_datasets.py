import os
import cv2 as cv
from glob import glob

VID_PATH = '/home/varun/PhD/datasets/VoxCeleb2/vox2_test_mp4/mp4'


# iterate through all the ids, videos and utterances
frame_count = 0
utt_count = 0
for idpath in sorted(glob(f'{VID_PATH}/*')):
    id = os.path.basename(idpath)
    if id in ['id08911', 'id04119', 'id07354']:  # val ids
        hrpath = '1_Train_deep_model/val/DIV2K_val_HR'
        lrpath = '1_Train_deep_model/val/DIV2K_val_LR_bicubic'
    else:
        hrpath = '1_Train_deep_model/train/DIV2K_train_HR'
        lrpath = '1_Train_deep_model/train/DIV2K_train_LR_bicubic/X4'
    for vpath in sorted(glob(f'{VID_PATH}/{id}/*')):
        vid = os.path.basename(vpath)
        for uttpath in sorted(glob(f'{VID_PATH}/{id}/{vid}/*')):
            utt = os.path.basename(uttpath)[:-4]
            # go through the video
            cap = cv.VideoCapture(uttpath)
            fc = 0
            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret:
                    if fc == 10:  # only taking one frame, the tenth frame
                        name = f'{id}_{vid}_{utt}_{fc}'
                        # write the HR image
                        cv.imwrite(f'{hrpath}/{name}_GT.png', frame)
                        # get the LR image and write
                        frame = cv.resize(frame, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv.INTER_CUBIC)
                        cv.imwrite(f'{lrpath}/{name}.png', frame)
                        # update counters
                        frame_count += 1
                        break
                    fc += 1
                else:
                    break
            utt_count += 1
            # When everything done, release the capture
            cap.release()
            cv.destroyAllWindows()
        print(f'{id}_{vid} done. Total utts: {utt_count} Total frames: {frame_count}')


