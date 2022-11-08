"""
This is for the case when we are finetuning the model for a particular face.
"""

import os
import cv2 as cv

# video for overfitting
id, vid, utt = 'id04119', '1uH67UruKlE', '00003'

# where I store the images for overfitting
lrvid = f'/home/varun/fvc/datasets/vox2_test_mp4_lr_x4/{id}/{vid}/{utt}.mp4'
hrvid = f'/home/varun/fvc/datasets/vox2_test_mp4/id04119/{id}/{vid}/{utt}.mp4'

# where I store the images for overfitting
lrpath = f'1_Train_deep_model/overfit/{id}/DIV2K_train_LR_bicubic/X4'
hrpath = f'1_Train_deep_model/overfit/{id}/DIV2K_train_HR'

if not os.path.exists(lrpath):
    os.makedirs(lrpath)
if not os.path.exists(hrpath):
    os.makedirs(hrpath)

# go through the video
for i, fpath in enumerate([lrvid, hrvid]):
    print(fpath)
    cap = cv.VideoCapture(fpath)
    fc = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        print(ret)
        if ret:
            # write the image
            if i == 0:
                cv.imwrite(f'{lrpath}/{fc}.png', frame)
                print(f'saving {fc} for lr')
            else:
                cv.imwrite(f'{hrpath}/{fc}_GT.png', frame)
                print(f'saving {fc} for hr')
            # update counters
            fc += 1
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
