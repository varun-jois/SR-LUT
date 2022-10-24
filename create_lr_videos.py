
import os
import glob

root_hr = '/home/varun/fvc/datasets/vox2_test_mp4'
root_lr = '/home/varun/fvc/datasets/vox2_test_mp4_lr_x4'

for n, iu in enumerate(glob.iglob(f'{root_hr}/*/*/*')):
    _, id, vid, utt = iu.rsplit('/', 3)
    vpath = os.path.join(root_lr, id, vid)
    # check if the dirs have been created already
    if not os.path.exists(vpath):
        os.makedirs(vpath)
    ou = os.path.join(vpath, utt)
    # check to see if the utt has been created
    if not os.path.exists(ou):
        # create the video using ffmpeg
        os.system(f'ffmpeg -i {iu} -vf scale=w=iw/4:h=ih/4 -c:v libx264 -preset veryfast {ou}')
    print(f'{n} done')
