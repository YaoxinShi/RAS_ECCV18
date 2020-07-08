import numpy as np
import scipy.misc
from PIL import Image
import scipy.io
import os
import cv2
import time

# Make sure that caffe is on the python path:
#caffe_root = 'C:\\Python3\\Lib\\site-packages\\caffe2\\'
#import sys
#sys.path.insert(0, caffe_root + 'python')

import caffe

EPSILON = 1e-8

data_root = './data/input/'
with open('./data/input/test.lst') as f:
    test_lst = f.readlines()
    
test_lst = [data_root+x.strip() for x in test_lst]

#remove the following two lines if testing with cpu
#caffe.set_mode_gpu()
# choose which GPU you want to use
#caffe.set_device(0)
caffe.SGDSolver.display = 0
# load net
net = caffe.Net('deploy.prototxt', 'ras_iter_10000.caffemodel', caffe.TEST)
save_root = './data/result/'
if not os.path.exists(save_root):
    os.mkdir(save_root)

start_time = time.time()
for idx in range(0, len(test_lst)):
    # load image
    print(test_lst[idx], "start")
    img = Image.open(test_lst[idx])
    img = np.array(img, dtype=np.uint8)
    im = np.array(img, dtype=np.float32)
    im = im[:,:,::-1]
    im -= np.array((104.00698793,116.66876762,122.67891434))
    im = im.transpose((2,0,1))
    
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *im.shape)
    net.blobs['data'].data[...] = im
    
    # run net and take argmax for prediction
    net.forward()
    res = net.blobs['sigmoid-score1'].data[0][0,:,:]
    
   # normalization
    res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
    res = 255*res;
    (filepath,tempfilename) = os.path.split(test_lst[idx])
    (filename,extension) = os.path.splitext(tempfilename)
    cv2.imwrite(save_root + filename + '.png', res)
    print(save_root + filename + '.png', "end")

diff_time = time.time() - start_time
print( 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst)))