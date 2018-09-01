import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys  
import caffe

images_name = []
with open('../data/tgs/predict/filelist.txt') as file:
    for line in file.readlines():
        line = line.strip('\n')
        images_name.append(line)

# load net
net = caffe.Net('deploy.prototxt', 'snapshot/train_iter_100000.caffemodel', caffe.TEST)

number = 0
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
for image_name in images_name:
    number += 1
    print('number:' + str(number))
    im = Image.open('../data/tgs/predict/images/' + image_name)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((125,0,100))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    #plt.imshow(out,cmap='gray');
    plt.imshow(out)
    plt.axis('off')
    plt.savefig('../data/tgs/predict/out/' + image_name)
    sem_out_img = Image.fromarray(out.astype('uint8')).convert('RGB')
    sem_out_img.save('../data/tgs/predict/masksout/' + image_name)

print('finish!')


