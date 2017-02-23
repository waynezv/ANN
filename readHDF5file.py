#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

file_path = './'
file_name = 'resolution_distorsion_simu_dataset_iq.hdf5'

with h5.File(file_path + file_name, 'r') as hf:
    for keys in hf.keys():
        print(keys)
    data = hf.get('US/US_DATASET0000')
    #  print(hf.attrs)
    #  print(data.attrs)
    #  print(hf.items)
    #  print(data.items)
    for items in data.items():
        print(items)
    np_data_real = np.array(data.get('data').get('real'))
    print(np_data_real.shape)
    #  print(np_data_real[0])
    np_data_imag = np.array(data.get('data').get('imag'))
    print(np_data_imag.shape)
    #  print(np_data_imag[0])
    #  print(data.items())
    #  np_data = np.array(data.get('data'))
    #  print(np_data.shape)
    s_np_dat_r = np_data_real[:, 1, 1]
    s_np_dat_i = np_data_imag[:, 1, 1]
    amp = np.sqrt(np.square(s_np_dat_r) + np.square(s_np_dat_i))

    plt.figure()
    plt.plot(np.arange(amp.shape[0]), amp)
    plt.show()
