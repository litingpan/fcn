import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import sys   
import caffe

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('coast_bea14.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy.prototxt', 'snapshot/train_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
sem_out = net.blobs['score_sem'].data[0].argmax(axis=0)
   
# plt.imshow(out,cmap='gray');
plt.imshow(sem_out)
plt.axis('off')
plt.savefig('coast_bea14_sem_out.png')
sem_out_img = Image.fromarray(sem_out.astype('uint8')).convert('RGB')
sem_out_img.save('coast_bea14_sem_img_out.png')

geo_out = net.blobs['score_geo'].data[0].argmax(axis=0)
plt.imshow(geo_out)
plt.axis('off')
plt.savefig('coast_bea14_geo_out.png')
geo_out_img = Image.fromarray(geo_out.astype('uint8')).convert('RGB')
geo_out_img.save('coast_bea14_geo_img_out.png')
