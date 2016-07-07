#!/usr/bin/env python

#  from acoular import __file__ as bpath, WNoiseGenerator, PointSource,\
#  Mixer, WriteH5, TimeSamples, PowerSpectra

#  from acoular import L_p, Calib, MicGeom, PowerSpectra, EigSpectra, \
    #  RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
    #  TimeSamples, MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
    #  TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
    #  BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
    #  BeamformerFunctional, WriteH5
#  from acoular.internal import digest

#  from traits.api import Float, Int, Property, Trait, Delegate, \
    #  cached_property, Tuple, HasPrivateTraits, CLong, File, Instance, Any, \
    #  on_trait_change, List, CArray
#  from traitsui.api import View, Item
#  from traitsui.menu import OKCancelButtons
#  import tables

from keras.models import Sequential, model_from_json, model_from_yaml
from keras.layers import Dense, Dropout, Activation, Merge, Flatten, \
    Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from os import path
import traceback
import numpy as np
from scipy import signal, fft, ifft
import matplotlib.pyplot as plt
from pylab import figure, plot, subplot, show, imshow, colorbar, axis, title
import h5py as h5


# Configurations
configurations = ('simu', 'expr')
metrics = ('resol', 'contr')
ftypes = ('iq', 'rf')

config = configurations[0]
metric = metrics[0]
ftype  = ftypes[1]

#  FREQ_C = 5208000.0
FREQ_S = 20832000.0

# Import Settings
if config == 'simu':
    phantom_path = './archive_to_download/database/simulation/resolution_distorsion/'
    phantom_name = 'resolution_distorsion_simu_phantom.hdf5'
    if metric == 'resol':
        data_path = './archive_to_download/database/simulation/resolution_distorsion/'
        imag_recon_path = './archive_to_download/reconstructed_image/simulation/resolution_distorsion/'
        if ftype == 'iq':
            data_name = 'resolution_distorsion_simu_dataset_iq.hdf5' # iq data
            imag_recon_name = 'resolution_distorsion_simu_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'resolution_distorsion_simu_dataset_rf.hdf5' # rf data
            imag_recon_name = 'resolution_distorsion_simu_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/simulation/resolution_distorsion/'
        scan_name = 'resolution_distorsion_simu_scan.hdf5'

    elif metric == 'contr':
        data_path = './archive_to_download/database/simulation/contrast_speckle/'
        imag_recon_path = 'archive_to_download/reconstructed_image/simulation/contrast_speckle/'
        if ftype == 'iq':
            data_name = 'contrast_speckle_simu_dataset_iq.hdf5' # iq data
            imag_recon_name = 'contrast_speckle_simu_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'contrast_speckle_simu_dataset_rf.hdf5' # if data
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
            data_name = 'resolution_distorsion_expe_dataset_iq.hdf5' # iq data
            imag_recon_name = 'resolution_distorsion_expe_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'resolution_distorsion_expe_dataset_rf.hdf5' # rf data
            imag_recon_name = 'resolution_distorsion_expe_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/experiments/resolution_distorsion/'
        scan_name = 'resolution_distorsion_expe_scan.hdf5'

    elif metric == 'contr':
        data_path = './archive_to_download/database/experiments/contrast_speckle/'
        imag_recon_path = 'archive_to_download/reconstructed_image/experiments/contrast_speckle/'
        if ftype == 'iq':
            data_name = 'contrast_speckle_expe_dataset_iq.hdf5' # iq data
            imag_recon_name = 'contrast_speckle_expe_img_from_iq.hdf5'
        elif ftype == 'rf':
            data_name = 'contrast_speckle_expe_dataset_rf.hdf5' # if data
            imag_recon_name = 'contrast_speckle_expe_img_from_rf.hdf5'

        scan_path = './archive_to_download/database/experiments/contrast_speckle/'
        scan_name = 'contrast_speckle_expe_scan.hdf5'

'''
# Time Sample Class
class iTimeSamples( TimeSamples ):
    """
    Container for time data in *.h5 format

    This class loads measured data from h5 files and
    and provides information about this data.
    It also serves as an interface where the data can be accessed
    (e.g. for use in a block chain) via the :meth:`result` generator.
    """

    @on_trait_change('basename')
    def load_data( self ):
        #""" open the .h5 file and set attributes
        #"""
        if not path.isfile(self.name):
            # no file there
            self.numsamples = 0
            self.numchannels = 0
            self.sample_freq = 0
            raise IOError("No such file: %s" % self.name)
        if self.h5f != None:
            try:
                self.h5f.close()
            except IOError:
                pass
        self.h5f = tables.open_file(self.name)
        self.data = self.h5f.root.time_data
        # DEBUG
        self.sample_freq = FREQ_S
        (self.numsamples, self.numchannels) = self.data.shape
'''


# Data Class
class DataSet:
    #  data = {} # storing the data
    #  scan = {}

    def __init__(self, name):
        # TODO: better initialize
        self.name = name
        self.data = {}
        self.probe_geom = ()
        self.fs = 0
        self.fm = 0
        self.c0 = 0
        self.num_chn = 0
        self.csm = () # cross spectral matrix
        self.csm_p = {} # cross spectral matrix splited in real and imag parts

        self.scan = {}

        self.pht_pos = ()

        self.scat_pos = ()


    def __print_name(self, name):
        print(name)

    def import_data(self, file_path, file_name):
        # TODO
        # try:
        assert path.exists(file_path+file_name), 'File not found.'
        with h5.File(file_path + file_name, 'r') as hf:
            print('This %s dataset contains: ' % file_name)
            hf.visit(self.__print_name)
            print

            hf_group = hf['/US/US_DATASET0000']

            # for received data
            if 'probe_geometry' in hf_group.keys():
                self.probe_geom = np.array(hf_group['probe_geometry'])
            if 'sampling_frequency' in hf_group.keys():
                self.fs = np.array(hf_group['sampling_frequency'])
            if 'modulation_frequency' in hf_group.keys():
                self.fm = np.array(hf_group['modulation_frequency'])
            if 'sound_speed' in hf_group.keys():
                self.c0 = np.array(hf_group['sound_speed'])
            if 'data' in hf_group.keys(): # also exists in reconstructed image data
                self.data['real'] = np.array(hf_group['data/real'])
                self.data['imag'] = np.array(hf_group['data/imag'])

            # for scan data
            if 'scan' in hf_group.keys():
                self.scan['x_axis'] = np.array(hf_group['scan/x_axis'])
                self.scan['z_axis'] = np.array(hf_group['scan/z_axis'])

            # for phantom data
            if 'phantom_xPts' in hf_group.keys():
                posx = np.array(hf_group['phantom_xPts'])
                posz = np.array(hf_group['phantom_zPts'])
                self.pht_pos = (posx, posz)

            if 'scatterers_positions' in hf_group.keys():
                self.scat_pos = np.array(hf_group['scatterers_positions'])

        # except IOError, e:
                # print(IOError, ':', e)

    def preprocess(self):
        # Cross spectral matrix
        (num_angles, num_channels, num_samples) = self.data['real'].shape
        nFFT = 512
        # (samples, channels, rows, cols)
        self.csm = np.zeros((num_angles, nFFT, num_channels, num_channels), dtype=complex)
        for k in np.arange(num_angles):
            for i in np.arange(num_channels):
                for j in np.arange(num_channels):
                    s1 = self.data['real'][k,i,:] + 1j*self.data['imag'][k,i,:]
                    s2 = self.data['real'][k,j,:] + 1j*self.data['imag'][k,j,:]
                    _, self.csm[k,:,i,j] = signal.csd(s1, s2, fs=FREQ_S, nperseg=40, \
                                    nfft=nFFT, scaling='density')

        # TODO
        # Diagnal removal
        #  diag_ind = (np.arange(num_channels), np.arange(num_channels))
        #  self.csm[diag_ind, :] = 0 + 1j*0

        # Lower triangle trim
        # Normalize: /Gxx Gyy


        print(self.csm.shape)
        self.csm_p['real'] = self.csm.real
        self.csm_p['imag'] = self.csm.imag

    def write_data(self, filename, channel_id):
        with h5.File(filename, 'w') as hf:
            # DEBUG: complex value OR absolute value ??
            (num_angles, num_channels, num_samples) = self.data['real'].shape
            one_ch_data = np.sqrt(self.data['real'][channel_id, :, :]**2 \
                    + self.data['imag'][channel_id, :, :]**2)
            #  mul_ch_data = np.sqrt( \
                    #  self.data['real'].reshape(num_angles*num_channels, num_samples)**2 \
                    #  + self.data['imag'].reshape(num_angles*num_channels, num_samples)**2 \
                    #  )

            hf['time_data'] = one_ch_data.T
            #  hf['time_data'] = mul_ch_data.T


    def show_image(self, prange):
        num_slices = self.data['real'].shape[0]
        plt.figure()
        for i in np.arange(num_slices):
            amp = np.sqrt(self.data['real'][i, :, :]**2 + self.data['imag'][i, :, :]**2)
            plt.subplot(2, 2, i+1)
            plt.imshow(amp, extent=prange)
            plt.title(i+1)
        plt.show()


class ANN(object):

    """Docstring for ANN. """

    def __init__(self):
        self.in_real = ()
        self.in_imag = ()
        #  self.input_data = ()

        self.out_real = ()
        self.out_imag = ()
        #  self.output_data = ()

    def train_mlp(self, input, output):
        self.in_real = input.data['real']
        self.in_imag = input.data['imag']
        self.out_real = output.data['real']
        self.out_imag = output.data['imag']

        (i_dim_x, i_dim_y, i_dim_z) = self.in_real.shape
        in_dim = i_dim_x*i_dim_y*i_dim_z
        input_data = self.in_real.reshape(in_dim, 1)

        (o_dim_x, o_dim_y, o_dim_z) = self.out_real.shape
        out_dim = o_dim_x*o_dim_y*o_dim_z
        output_data = self.out_real.reshape(out_dim, 1)

        model = Sequential()
        model.add(Dense(200, input_dim=in_dim, init='uniform'))
        model.add(Activation('relu'))
        #  model.add(Dropout(0.25))

        model.add(Dense(200))#, init='uniform'))
        model.add(Activation('relu'))
        #  model.add(Dropout(0.25))

        model.add(Dense(out_dim))#, init='uniform'))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='sgd',\
                metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        hist = model.fit(input_data, output_data, nb_epoch=50, \
                         batch_size=64, validation_split=0.2, \
                         shuffle=True, callbacks=[early_stop])
        print(hist.history)
        #TODO: batch train
        model.train_on_batch()

        # Save model
        model_to_save_json = model.to_json()
        open('model_architecture.json', 'w').write(model_to_save_json)
        model_to_save_yaml = model.to_yaml()
        open('model_architecture.yaml', 'w').write(model_to_save_yaml)
        model.save_weights('weights.h5')

    def train_cnn(self, input, output):
        self.in_real = input.csm_p['real']
        self.in_imag = input.csm_p['imag']
        self.out_real = output.data['real']
        self.out_imag = output.data['imag']

        (num_samples, num_channels, num_rows, num_cols) = self.in_real.shape
        (o_dim_x, o_dim_y, o_dim_z) = self.out_real.shape
        out_dim = o_dim_x*o_dim_y*o_dim_z
        out_data_r = self.out_real.reshape(out_dim, 1)
        out_data_i = self.out_imag.reshape(out_dim, 1)

        batch_size = 32
        num_epoch = 200
        num_filter = 64
        num_row_kernel = 3
        num_col_kernel = 3
        num_pool = 2
        dim_order = 'th' # (samples, channels, rows, cols)
        model = Sequential()
        model.add(Convolution2D(num_filter, num_row_kernel, num_col_kernel, \
                                border_mode='same', dim_ordering=dim_order, \
                                input_shape=(num_channels, num_rows, num_cols)))
        model.add(Activation('relu'))
        model.add(Convolution2D(num_filter, num_row_kernel, num_col_kernel))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(num_pool, num_pool)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(num_filter, num_row_kernel, num_col_kernel, \
                                border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(num_filter, num_row_kernel, num_col_kernel))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(num_pool, num_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(out_dim))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', \
                      optimizer=sgd, metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        hist_r = model.fit(self.in_real, out_data_r, \
                  bash_size=batch_size, nb_epoch=num_epoch, verbose=1, \
                  validation_split=0.2, shuffle=True, callbacks=[early_stop])
        print(hist_r.history)

        hist_i = model.fit(self.in_imag, out_data_i, \
                  bash_size=batch_size, nb_epoch=num_epoch, verbose=1, \
                  validation_split=0.2, shuffle=True, callbacks=[early_stop])
        print(hist_i.history)

    def predict(self, X_test, Y_test):

        model = model_from_json(open('model_architecture.json').read())
        model = model_from_yaml(open('model_architecture.yaml').read())
        model.load_weights('weights.h5')
        loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

        #  classes = model.predict_classes(X_test, batch_size=32)

        proba = model.predict_proba(X_test, batch_size=32)

    def get_interlayer_output(self, num_layer):
        """TODO: Docstring for get_interlayer_output.
        :returns: TODO

        """
        pass


# TODO: remove
def test_import():
    rcv_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'data')
    rcv_data.import_data(data_path, data_name)
    rcv_data.preprocess()

    #  sc_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'scan')
    #  sc_data.import_data(scan_path, scan_name)

    #  img_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'reconstructed_image')
    #  img_data.import_data(imag_recon_path, imag_recon_name)

    #  pht_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'pht_data')
    #  pht_data.import_data(phantom_path, phantom_name)

def test_net():
    rcv_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'data')
    rcv_data.import_data(data_path, data_name)
    rcv_data.preprocess()

    img_data = DataSet('reconstructed_image')
    img_data.import_data(imag_recon_path, imag_recon_name)

    ann = ANN()
    #  ann.train_mlp(rcv_data, img_data)
    ann.train_cnn(rcv_data, img_data)

def beamform_imaging():
    rcv_data = DataSet(config+'_'+metric+'_'+ftype+'_'+'data')
    rcv_data.import_data(data_path, data_name)
    pw_id = np.arange(75)
    print(pw_id)
    for i in pw_id:
        rcv_data_name = 'one_ch_samples_'
        rcv_data_name_id = rcv_data_name + str(i) + '.h5'
        rcv_data.write_data(rcv_data_name_id, i)

    #  mg = MicGeom(from_file='rx_geom.xml')
    #  rg = RectGrid(x_min=-0.8, x_max=-0.2, y_min=-0.1, y_max=0.3, z=0.8,\
    #  increment=0.01)
    #  t1 = iTimeSamples(name=rcv_data_name, numchannels=128)
    #  #  cal = Calib(from_file='calibration_data.xml')
    #  f1 = EigSpectra(time_data=t1, block_size=256, window="Hanning",\
            #  overlap='75%')
            #  #  calib=cal)
    #  e1 = BeamformerBase(freq_data=f1, grid=rg, mpos=mg, r_diag=False)
    #  # TODO: fr ?
    #  fr = 100000
    #  fr = 500000
#  #    fr = 600000
    #  fr = 700000
#  #    fr = 800000
#  #    fr = 900000
    #  pm = e1.synthetic(fr, 0)
    #  Lm = L_p(pm)
    #  imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), \
        #  interpolation='bicubic')

    #  colorbar()

    #  show()

# analyze the data and generate map
    #  ts = TimeSamples( name=h5savefile )
    #  ps = PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
    #  rg = RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
    #  increment=0.01 )
    #  bb = BeamformerBase( freq_data=ps, grid=rg, mpos=mg )
    #  pm = bb.synthetic( 8000, 3 )
    #  Lm = L_p( pm )

# show map
    #  imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), \
    #  interpolation='bicubic')
    #  colorbar()


if __name__ == '__main__':
    test_net()
    #  test_import();
    #  beamform_imaging()
