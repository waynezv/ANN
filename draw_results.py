#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import argparse as agp

from ANN_large_v2 import nice_show

parser = agp.ArgumentParser(description = "Select input:")
parser.add_argument('-i', '--input', \
        help = 'input filename', required = True)
args = parser.parse_args()

predname = args.input

num_samples = 82
label_size = 32

#  labelname = 'output_h5_n1000_59s'
labelname = 'output_h5_n2000_82s'
with h5.File(labelname, 'r') as hf:
    label = np.array(hf['output'])

with h5.File(predname, 'r') as hf:
    pred = np.array(hf['pred'])
#nS,nR = pred.shape
#pred = pred.reshape(nS,label_size,label_size)

print(pred.shape)
pred = pred.reshape(-1,32,32)

imgl = np.zeros((num_samples,label_size,label_size))
imgp = np.zeros((num_samples,label_size,label_size))
for i in range(num_samples):
    imgl[i,:,:] = label[i,:].reshape(label_size,label_size)
    imgp[i,:,:] = pred[i,:,:]

f = plt.figure()
nice_show(f,imgl)
f = plt.figure()
nice_show(f,imgp)
plt.show()
