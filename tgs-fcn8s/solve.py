import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../tgs-fcn16s/snapshot/train_iter_100000.caffemodel'

# init
# caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)


# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/tgs/dataset/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    score.seg_tests(solver, False, test, layer='score')
