import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys  
import caffe

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# im = Image.open('../data/tgs/predict/images/000c8dfb2a.png')
im = Image.open('../data/tgs/dataset/images/0a1742c740.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((125,0,100))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy.prototxt', 'snapshot/train_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
sem_out = net.blobs['score'].data[0].argmax(axis=0)
 
#plt.imshow(out,cmap='gray');
plt.imshow(sem_out)
plt.axis('off')
plt.savefig('test_out.png')
sem_out_img = Image.fromarray(sem_out.astype('uint8')).convert('RGB')
sem_out_img.save('test_img_out.png')


