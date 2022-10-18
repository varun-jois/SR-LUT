from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
import cv2 as cv

def lut_sr(img):
    LUT_PATH = "Model_S_x{}_{}bit_int8.npy".format(UPSCALE, SAMPLING_INTERVAL)


