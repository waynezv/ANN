#!/usr/bin/env python
# encoding: utf-8

import numpy as np
#  import matplotlib.pyplot as plt
import h5py as h5

from keras.models import Sequential
from keras.layers import Dense, Activation, Merge

configurations = ('simu', 'expr')
metrics        = ('resol', 'contr')
ftypes         = ('iq', 'rf')

config = configurations[0]
metric = metrics[0]
ftype  = ftypes[0]

if config == 'simu':
    phantom_path = './archive_to_download/database/simulation/resolution_distorsion/'
    phantom_name = 'resolution_distorsion_simu_phantom.hdf5'
    if metric == 'resol':
        data_path = './archive_to_download/database/simulation/resolution_distorsion/'
        if ftype == 'iq':
            data_name = 'resolution_distorsion_simu_dataset_iq.hdf5' # IQ data
        elif ftype == 'rf':
            data_name = 'resolution_distorsion_simu_dataset_rf.hdf5' # RF data

        scan_path = './archive_to_download/database/simulation/resolution_distorsion/'
        scan_name = 'resolution_distorsion_simu_scan.hdf5'

    elif metric == 'contr':
        data_path = './archive_to_download/database/simulation/contrast_speckle/'
        if ftype == 'iq':
            data_name = 'contrast_speckle_simu_dataset_iq.hdf5' # IQ data
        elif ftype == 'rf':
            data_name = 'contrast_speckle_simu_dataset_rf.hdf5' # IQ data

        scan_path = './archive_to_download/database/simulation/contrast_speckle/'
        scan_name = 'contrast_speckle_simu_scan.hdf5'

elif config == 'expr':
    phantom_path = './archive_to_download/database/experiments/resolution_distorsion/'
    phantom_name = 'resolution_distorsion_expe_phantom.hdf5'
    if metric == 'resol':
        data_path = './archive_to_download/database/experiments/resolution_distorsion/'
        if ftype == 'iq':
            data_name = 'resolution_distorsion_expe_dataset_iq.hdf5' # IQ data
        elif ftype == 'rf':
            data_name = 'resolution_distorsion_expe_dataset_rf.hdf5' # RF data

        scan_path = './archive_to_download/database/experiments/resolution_distorsion/'
        scan_name = 'resolution_distorsion_expe_scan.hdf5'

    elif metric == 'contr':
        data_path = './archive_to_download/database/experiments/contrast_speckle/'
        if ftype == 'iq':
            data_name = 'contrast_speckle_expe_dataset_iq.hdf5' # IQ data
        elif ftype == 'rf':
            data_name = 'contrast_speckle_expe_dataset_rf.hdf5' # IQ data

        scan_path = './archive_to_download/database/experiments/contrast_speckle/'
        scan_name = 'contrast_speckle_expe_scan.hdf5'

class DataSet:
    data_set = {};
    def __init__(self, name):
        self.name = name


def print_name(name):
    print(name)

def import_data(file_path, file_name):
    with h5.File(file_path + file_name, 'r') as hf:

        print('This %s dataset contains: ' % file_name)
        hf.visit(print_name)
        print

        data = hf['/US/US_DATASET0000']

        '''
        print('Items:')
        for item in data.items():
            print(item)
        print
        '''

        data_obj = DataSet(file_name)

        print('Getting keys and values:')
        for key in data.keys():
            print key
            #  print data[key]
            data_obj.data_set[key] = data[key]
            print data_obj.data_set.get(key)
            print
        #  for name in data:
            #  print(name)
        print

        #  print(data_obj.data_set['data'])

        #  np_data_real = np.array(data_obj.data_set['data']['real'])
        #  #  np_data_real = np.array(data['data/real'])
        #  print(np_data_real.shape)
        #  print(np_data_real.size)
        #  print(np_data_real.dtype)

        '''
        np_data_imag = np.array(data['data/imag'])
        print(np_data_imag.shape)
        print(np_data_imag.size)
        print(np_data_imag.dtype)

        s_np_dat_r = np_data_real[:, 1, 1]
        s_np_dat_i = np_data_imag[:, 1, 1]
        amp = np.sqrt(np.square(s_np_dat_r) + np.square(s_np_dat_i))

        #  PLOT
        plt.figure()
        plt.plot(np.arange(amp.size), amp)
        plt.plot(np.arange(amp.size), amp, 'ro')
        plt.show()
        '''

    return data_obj

def test():
    import_data(data_path, data_name)
    import_data(scan_path, scan_name)
    import_data(phantom_path, phantom_name)

'''
def train(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(ouput_dim = , input_dim = ))
    model.add(Activation('relu'))
    model.add(Dense(output_dim = ))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd',\
            metrics=['accuracy'])
    hist = model.fit(X_train, Y_train, nb_epoch=5, batch_size=32,\
            validation_split=0.1, shuffle=True)
    print(hist.history)
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

    classes = model.predict_classes(X_test, batch_size=32)
    proba = model.predict_proba(X_test, batch_size=32)
'''

if __name__ == '__main__':
    test()
