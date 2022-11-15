"""
This is for the case when we are finetuning the model for a particular face. Here I prepare training
data as well as the testing data.
"""

import os
import cv2 as cv

# video for overfitting
id = 'id04119'
vids_utts = {'1uH67UruKlE': ['00003']}

# video for testing
val_vids_utts = {'1uH67UruKlE': ['00002']}

# go through the video
for dtype in ['train', 'test']:
    if dtype == 'train':
        data = vids_utts
        # where I store the images
        lrpath = f'1_Train_deep_model/overfit/{id}/DIV2K_train_LR_bicubic/X4'
        hrpath = f'1_Train_deep_model/overfit/{id}/DIV2K_train_HR'
    else:
        data = val_vids_utts
        lrpath = f'1_Train_deep_model/overfit/{id}/val_LR'
        hrpath = f'1_Train_deep_model/overfit/{id}/val_HR'

    if not os.path.exists(lrpath):
        os.makedirs(lrpath)
    if not os.path.exists(hrpath):
        os.makedirs(hrpath)

    for vid in data:
        for utt in data[vid]:
            lrvid = f'/home/varun/fvc/datasets/vox2_test_mp4_lr_x4/{id}/{vid}/{utt}.mp4'
            hrvid = f'/home/varun/fvc/datasets/vox2_test_mp4/{id}/{vid}/{utt}.mp4'

            caplr = cv.VideoCapture(lrvid)
            caphr = cv.VideoCapture(hrvid)

            fc = 0
            while True:
                # Capture frame-by-frame
                retlr, framelr = caplr.read()
                rethr, framehr = caphr.read()
                if retlr and rethr:
                    # write the image
                    cv.imwrite(f'{lrpath}/{fc}.png', framelr)
                    cv.imwrite(f'{hrpath}/{fc}_GT.png', framehr)
                    # update counters
                    fc += 1
                else:
                    break
            # When everything done, release the capture
            caplr.release()
            caphr.release()
            cv.destroyAllWindows()
