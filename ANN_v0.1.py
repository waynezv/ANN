#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge

# Configurations
configurations = ('simu', 'expr')
metrics        = ('resol', 'contr')
ftypes         = ('iq', 'rf')

config = configurations[0]
metric = metrics[0]
ftype  = ftypes[0]

# Import Settings
if config == 'simu':
    phantom_path = './archive_to_download/database/simulation/resolution_distorsion/'
    phantom_name = 'resolution_distorsion_simu_phantom.hdf5'
    if metric == 'resol':
        data_path = './archive_to_download/database/simulation/resolution_distorsion/'
        imag_recon_path = './archive_to_download/reconstructed_image/simulation/resolution_distorsion/'
        if ftype == 'iq':
            data_name = 'resolution_distorsion_simu_dataset_iq.hdf5' # IQ data
            imag_recon_name = 'resolution_distorsion_simu_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'resolution_distorsion_simu_dataset_rf.hdf5' # RF data
            imag_recon_name = 'resolution_distorsion_simu_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/simulation/resolution_distorsion/'
        scan_name = 'resolution_distorsion_simu_scan.hdf5'

    elif metric == 'contr':
        data_path = './archive_to_download/database/simulation/contrast_speckle/'
        imag_recon_path = 'archive_to_download/reconstructed_image/simulation/contrast_speckle/'
        if ftype == 'iq':
            data_name = 'contrast_speckle_simu_dataset_iq.hdf5' # IQ data
            imag_recon_name = 'contrast_speckle_simu_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'contrast_speckle_simu_dataset_rf.hdf5' # IQ data
            imag_recon_name = 'contrast_speckle_simu_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/simulation/contrast_speckle/'
        scan_name = 'contrast_speckle_simu_scan.hdf5'

elif config == 'expr':
    phantom_path = './archive_to_download/database/experiments/resolution_distorsion/'
    phantom_name = 'resolution_distorsion_expe_phantom.hdf5'
    if metric == 'resol':
        data_path = './archive_to_download/database/experiments/resolution_distorsion/'
        imag_recon_path = 'archive_to_download/reconstructed_image/experiments/resolution_distorsion/'
        if ftype == 'iq':
            data_name = 'resolution_distorsion_expe_dataset_iq.hdf5' # IQ data
            imag_recon_name = 'resolution_distorsion_expe_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'resolution_distorsion_expe_dataset_rf.hdf5' # RF data
            imag_recon_name = 'resolution_distorsion_expe_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/experiments/resolution_distorsion/'
        scan_name = 'resolution_distorsion_expe_scan.hdf5'

    elif metric == 'contr':
        data_path = './archive_to_download/database/experiments/contrast_speckle/'
        imag_recon_path = 'archive_to_download/reconstructed_image/experiments/contrast_speckle/'
        if ftype == 'iq':
            data_name = 'contrast_speckle_expe_dataset_iq.hdf5' # IQ data
            imag_recon_name = 'contrast_speckle_expe_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'contrast_speckle_expe_dataset_rf.hdf5' # IQ data
            imag_recon_name = 'contrast_speckle_expe_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/experiments/contrast_speckle/'
        scan_name = 'contrast_speckle_expe_scan.hdf5'


# Data Class
class DataSet:
    #  data = {} # storing the data
    #  scan = {}

    def __init__(self, name):
        self.name = name
        self.data = {}
        self.scan = {}


    def __print_name(self, name):
        print(name)

    def import_data(self, file_path, file_name):
        with h5.File(file_path + file_name, 'r') as hf:
            print('This %s dataset contains: ' % file_name)
            hf.visit(self.__print_name)
            print

            hf_group = hf['/US/US_DATASET0000']
            #  for key in hf_group:
                #  print(hf_group[key])
                #  tmp = np.array(hf_group[key])
                #  print(tmp)
            if 'data' in hf_group.keys():
                self.data['real'] = np.array(hf_group['data/real'])
                self.data['imag'] = np.array(hf_group['data/imag'])
            if 'scan' in hf_group.keys():
                self.scan['x_axis'] = np.array(hf_group['scan/x_axis'])
                self.scan['z_axis'] = np.array(hf_group['scan/z_axis'])
            if 'phantom_xPts' in hf_group.keys():
                posx = np.array(hf_group['phantom_xPts'])
                posz = np.array(hf_group['phantom_zPts'])
                pos = (posx, posz)
                print(pos)
                print
                print(pos[0])
                print
                print(pos[1])
                plt.figure()
                plt.plot(posx, posz, 's')
                plt.title('1')
                plt.show()
            if 'scatterers_positions' in hf_group.keys():
                pos = np.array(hf_group['scatterers_positions'])
                print(pos.shape)
                plt.figure()
                plt.plot(pos[0], pos[2], 's')
                plt.title('2')
                plt.show()

    def show_image(self, prange):
        num_slices = self.data['real'].shape[0]
        plt.figure()
        for i in np.arange(num_slices):
            amp = np.sqrt(self.data['real'][i, :, :]**2 + self.data['imag'][i, :, :]**2)
            plt.subplot(2, 2, i+1)
            plt.imshow(amp, extent=prange)
            plt.title(i+1)
        plt.show()

def train(in_dim, out_dim, X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(100000, input_dim = in_dim, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100000, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(out_dim, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd',\
            metrics=['accuracy'])

    hist = model.fit(X_train, Y_train, nb_epoch=5, batch_size=32,\
            validation_split=0.1, shuffle=True)
    print(hist.history)

    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

    classes = model.predict_classes(X_test, batch_size=32)

    proba = model.predict_proba(X_test, batch_size=32)

def test_import():
    pht_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'pht_data')
    pht_data.import_data(phantom_path, phantom_name)

def test_net():
    rcv_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'data')
    rcv_data.import_data(data_path, data_name)
    #  import_data(scan_path, scan_name)
    #  import_data(phantom_path, phantom_name)
    img_data = DataSet('reconstructed_image')
    img_data.import_data(imag_recon_path, imag_recon_name)
    #  img_data.show_image([0, 0.1, 0, 0.1])

    input_dim = rcv_data.data['real'].size
    print input_dim
    X_train = rcv_data.data['real'].reshape(1, -1)

    output_dim = img_data.data['real'].size
    print output_dim
    Y_train = img_data.data['real'].reshape(1, -1)

    X_test = rcv_data.data['imag'].reshape(1, -1)
    print X_test.shape

    Y_test = img_data.data['imag'].reshape(1, -1)
    print Y_test.shape

    #  train(input_dim, output_dim, X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    test_net()
    #  test_import();
