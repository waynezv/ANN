#!/usr/bin/env python

from keras.models import Sequential, model_from_json, model_from_yaml
from keras.layers import Dense, Dropout, Activation, Merge, Flatten, \
    Convolution2D, MaxPooling2D, ZeroPadding2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import backend as K
#  from keras.utils.visualize_util import plot as kplt

import theano as T

from os import path
import traceback
import numpy as np
from numpy import matlib
from scipy import signal, fft, ifft
from scipy.fftpack import dct, idct
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import figure, plot, subplot, show, imshow, colorbar, axis, title
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import h5py as h5
import re

# Data Class
class DataSet:
    #  data = {}
    #  scan = {}

    def __init__(self, name):
        # TODO: better initialize
        self.name = name
        self.data = {} # storing the data
        self.probe_geom = ()
        self.angles = ()
        self.fs = 0
        self.fm = 0
        self.c0 = 0
        self.num_chn = 0
        self.csm = () # cross spectral matrix

        self.scan = {}
        self.dist = ()

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

        # except IOError, e:
                # print(IOError, ':', e)

    def preprocess(self):
        # Cross spectral matrix
        mode = 1 # 0: without average, of shape (num_angles, nFFT, num_channels, num_channels)
                 # 1: with average, of shape (num_angles, num_channels, num_channels)

        (num_angles, num_channels, num_samples) = self.data['real'].shape
        nFFT = 512 # 256
        if mode == 0:
            # (samples, channels, rows, cols)
            self.csm = np.zeros((num_angles, nFFT, num_channels, num_channels), dtype=complex)
            for k in np.arange(num_angles):
                for i in np.arange(num_channels):
                    s1 = sef.data['real'][k,i,:] # ignore imaginary part
                    for j in np.arange(i+1,num_channels):
                        s2 = self.data['real'][k,j,:]
                        _, self.csm[k,:,j,i] = signal.csd(s1, s2, fs=FREQ_S, nperseg=nFFT, \
                                        nfft=nFFT, scaling='density')
        # TODO
        # Diagnal removal: use a better algorithm
        # lambda filter map reduce

        elif mode == 1:
            self.csm = np.zeros((num_angles, num_channels, num_channels), dtype=float)
            for k in np.arange(num_angles):
                for i in np.arange(num_channels):
                    s1 = self.data['real'][k,i,:]
                    for j in np.arange(i+1,num_channels):
                        s2 = self.data['real'][k,j,:]
                        _, tmp = signal.csd(s1, s2, fs=FREQ_S, nperseg=nFFT, \
                                nfft=nFFT, scaling='density')
                        self.csm[k,j,i] = np.abs(np.sum(tmp) / nFFT) # sum,average,abs
                    self.csm[k,i,(i+1):num_channels] = self.csm[k,(i+1):num_channels,i]
        print(self.csm.shape)
        print(self.csm)

        with h5.File('csm_h5', 'w') as hf:
            hf['csm'] = self.csm

        #  csm_t = self.csm[1,:,:]
        #  img = csm_t.reshape(num_channels, num_channels)
        #  plt.figure()
        #  plt.imshow(img)
        #  plt.show()

        # Lower triangle trim
        # Normalize: /Gxx Gyy

    def compute_dist(self):
        # Distance matrix
        num_x = len(self.scan['x_axis'])
        num_z = len(self.scan['z_axis'])
        self.dist = np.zeros((num_x, num_z), dtype=float)
        for i in range(num_x):
            for j in range(num_z):
                self.dist[i,j] = np.sqrt(self.scan['x_axis'][i]**2 \
                        + self.scan['z_axis'][j]**2)

    def write_data(self, filename, channel_id):
        with h5.File(filename, 'w') as hf:
            # DEBUG: complex value OR absolute value ??
            (num_angles, num_channels, num_samples) = self.data['real'].shape
            one_ch_data = np.sqrt(self.data['real'][channel_id, :, :]**2 \
                    + self.data['imag'][channel_id, :, :]**2)
            hf['time_data'] = one_ch_data.T

            #  mul_ch_data = np.sqrt( \
                    #  self.data['real'].reshape(num_angles*num_channels, num_samples)**2 \
                    #  + self.data['imag'].reshape(num_angles*num_channels, num_samples)**2 \
                    #  )
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


def img_norm(img):
    min = np.amin(img)
    max = np.amax(img)
    return (img-min) / (max-min)

def nice_show(fig, data, vmin=None, vmax=None, cmap=None):
    '''
    data is 3D (nCH, nCol, nRow)
    '''
    assert data.ndim==3, 'Data dimension must be 3!'
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    nCH,_,_= data.shape
    nr = int(np.ceil(np.sqrt(nCH)))
    assert nr<=10, 'Too many data channels (>10)!'
    grid = ImageGrid(fig, 111, \
            nrows_ncols=(nr, nr),\
            axes_pad=0.1,\
            add_all=True,\
            label_mode='L')
    for i in range(nCH):
        ax = grid[i]
        im = ax.imshow(data[i,:,:], vmin=vmin, vmax=vmax, \
                interpolation='nearest', cmap=cmap)
#    div = make_axes_locatable(ax)
#    cax = div.append_axes('right', size='5%', pad=0.05) # colorbar axis to the right
#    plt.colorbar(im, cax=cax)

class ANN(object):

    """Docstring for ANN. """

    def __init__(self):
        self.in_real = ()
        self.in_imag = ()

        self.out_real = ()
        self.out_imag = ()

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

    def train_cnn(self, input, output, test_input):
        num_samples, num_channels, num_rows, num_cols = input.shape
        output = output.reshape(num_samples,32*32)

        # Configurations
        batch_size = 100
        num_epoch = 500

        model = Sequential()
        '''
        model.add(Convolution2D(16,4,4, \
                subsample=(1,1), \
                activation='relu', \
                input_shape=(num_channels,num_rows,num_cols)))
        model.add(Convolution2D(16,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(16,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(16,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(16,4,4, \
                subsample=(1,1), \
                activation='relu'))

        model.add(Convolution2D(32,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(32,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(32,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(32,4,4, \
                subsample=(1,1), \
                activation='relu'))

        model.add(Convolution2D(64,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(64,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(64,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(64,4,4, \
                subsample=(1,1), \
                activation='relu'))

        model.add(Convolution2D(128,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(128,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(128,4,4, \
                subsample=(1,1), \
                activation='relu'))

        model.add(Convolution2D(256,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(256,4,4, \
                subsample=(1,1), \
                activation='relu'))

        model.add(Convolution2D(512,4,4, \
                subsample=(1,1), \
                activation='relu'))
        model.add(Convolution2D(512,4,4, \
                subsample=(1,1), \
                activation='relu'))

        model.add(Convolution2D(1024,4,4, \
                subsample=(1,1), \
                activation='relu'))
        '''
        #model.add(Flatten())
        #model.add(BatchNormalization(mode=0, axis=1))
        #model.add(Dropout(0.20))
        model.add(Reshape((num_channels*num_rows*num_cols,), \
                input_shape=(num_channels,num_rows,num_cols)))
        model.add(Dense(256, \
                #input_dim=num_channels*num_rows*num_cols, \
                activation='relu' ))
        model.add(Dense(256, \
                activation='relu' ))
        model.add(Dense(256, \
                activation='relu' ))
        model.add(Dense(128, \
                activation='relu' ))
        model.add(Dense(256, \
                activation='relu' ))
        model.add(Dense(256, \
                activation='sigmoid' ))
        model.add(Dense(32*32))

        # Compile
        sgd = SGD(lr=0.1, decay=0.09, momentum=0.0, nesterov=False)
        model.compile( optimizer='sgd', \
                loss='mean_squared_error' )

        #early_stop = EarlyStopping(monitor='val_loss', patience=2)
        early_stop = EarlyStopping(monitor='loss', patience=4)
        hist = model.fit(input, output, \
                  batch_size=batch_size, nb_epoch=num_epoch, verbose=1, \
                  shuffle=True, \
                  validation_split=0.1, \
                  callbacks=[early_stop])

        # TODO: move Prediction to a seperated func
        # Prediction
        eval = model.evaluate(input, output, batch_size=batch_size)
        print('eval', eval)
        #predict = model.predict(test_input, batch_size=batch_size)
        predict = model.predict(input, batch_size=batch_size)
        print('predict', predict)
        #  rmse = np.sqrt(((predict-output)**2).mean(axis=0))
        #  print("rmse = ")
        #  print(rmse)

        #  model.train_on_batch(self.in_real, out_data_r)
        #  model.train_on_batch(self.in_imag, out_data_i)

        # TODO: save model
        #model_to_save_json = model.to_json()
        #open('model_architecture.json', 'w').write(model_to_save_json)
        #model_to_save_yaml = model.to_yaml()
        #open('model_architecture.yaml', 'w').write(model_to_save_yaml)
        #model.save_weights('weights.h5')

# Visualization
        '''
        I1 = input
        print("I1 shape: ", I1.shape)
        print('layer 0: ', model.layers[0].get_config())
        print

        l1f = T.function([model.layers[0].input], \
                model.layers[1].output, allow_input_downcast=True)
        l1o = np.array(l1f(I1))
        print('layer 1: ', model.layers[1].get_config())
        print("l1o shape: ", l1o.shape)
        l1w = np.squeeze(model.layers[1].W.get_value(borrow=True))
        #  W1 = model.layers[1].get_weights()[0] # 0 is W, 1 is b
        print("l1w shape: ", l1w.shape)
        print

        l2f = T.function([model.layers[1].input], \
                act1.output, allow_input_downcast=True)
        l2o = np.array(l2f(I1))
        print('layer 2: ', model.layers[2].get_config())
        print("l2o shape: ", l2o.shape)
        print

        f = plt.figure()
        plt.title('I1')
        nice_show(f,I1[0])
        f = plt.figure()
        plt.title('l1w')
        nice_show(f,l1w[0])
        f = plt.figure()
        plt.title('l2o')
        nice_show(f,l2o[0])

        plt.show()
        '''

        return predict

    def predict(self, X_test, Y_test):

        model = model_from_json(open('model_architecture.json').read())
        model = model_from_yaml(open('model_architecture.yaml').read())
        model.load_weights('weights.h5')
        loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)

        classes = model.predict_classes(X_test, batch_size=32)

        proba = model.predict_proba(X_test, batch_size=32)

    def get_interlayer_output(self, num_layer):
        """TODO: Docstring for get_interlayer_output.
        :returns: TODO

        """
        pass



def get_2D_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def get_2D_idct(coeff):
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')

def test_net():
    nFFT = 512
    num_channels = nFFT//2+1
    num_rows = 64
    num_cols = 64
    label_size = 32
    dct_size = 25

    prefix = '/mingback/zhaowenbo'
    #input_name = 'csm_h5_n2000_1600s'
    input_name = 'csm_h5_n2000_1600s'
    output_name = 'outcat_h5_n2000_1600s'
    test_input_name ='input_h5_n2000_82s'
    test_output_name = 'output_h5_n2000_82s'
    pat = re.compile('(\w+)(\_\w+)(\_\w+\_)(\d+)(\w+)')
    mat = pat.split(input_name)
    num_samples = int(mat[4])
    print('num_samples', num_samples)
    mat = pat.split(test_input_name)
    num_tests = int(mat[4])
    print('num_tests', num_tests)

    print('Preparing inputs ...')
    with h5.File(path.join(prefix,input_name), 'r') as hf:
        #input_data = np.array(hf['input'])
        input_data = np.array(hf['csm'])
    print('Preparing outputs ...')
    with h5.File(path.join(prefix,output_name), 'r') as hf:
        output_data = np.array(hf['output'])
    #output_data = output_data.reshape(num_samples,label_size,label_size)

    with h5.File(test_input_name, 'r') as hf:
        test_input_data = np.array(hf['input'])
    with h5.File(test_output_name, 'r') as hf:
        test_output_data = np.array(hf['output'])
    #test_output_data = test_output_data.reshape(num_tests,label_size,label_size)


    # Train
    ann = ANN()
    pred = ann.train_cnn(input_data, output_data, \
            test_input_data)

    #parser = agp.ArgumentParser(description='Output:')
    #parser.add_argument('-o', '--output', \
    #        help='output filename', required=True)
    #args = parser.parse_args()
    #outputname = args.output
    with h5.File('pred_resul_h5_v3', 'w') as hf:
        hf['pred'] = pred

    '''
    images_pred = []
    for i in range(num_samples):
        dct_pr = pred[i,:]
        dct_pr_cp = np.zeros((label_size,label_size))
        dct_pr_cp[:dct_size,:dct_size] = dct_pr.copy()
        img_pr = get_2D_idct(dct_pr_cp)
        images_pred.append(img_pr)
    with h5.File('images_pred.h5', 'w') as hf:
        hf['images_pred'] = images_pred
    '''

    #  print('amp_pr is ')
    #  print(amp_pr)

    #  plt.figure()
    #  plt.imshow(amp_pr[0,:,:], extent=(0,0.1,0,0.1))
    #  plt.show()

def test_import():
    num_samples = 82
    num_channels = 66
    num_rows = 64
    num_cols = 64

    dataset_name = 'animals_n2000_s32_30-Jul-2016'
# Read Inputs
    prefix = ''.join(['/home1/zhaowenbo/Field_II_ver_3_24_linux/gen_samples/sim_data_', \
            dataset_name])
    input_data = np.zeros((num_samples, num_channels, num_rows, num_cols))
    #sam_ls = list( set(range(1,122))-set([16,90]) )
    sam_ls = range(1,83)
    '''
    for sid in range(num_samples):
        sam_id = sam_ls[sid]
        ln_id = 63
        for cid in range(num_channels):
            inname = ''.join(['phant_',str(sam_id),'_rf_ln',str(ln_id),'.mat'])
            print('inname', inname)
            input_data[sid,cid,:,:] = sio.loadmat(path.join(prefix,inname))['csm']
            ln_id += 1
    with h5.File('input_h5_n2000_82s', 'w') as hf:
        hf['input'] = input_data
    '''

# Read Outputs
    label_data_path = '/home1/zhaowenbo/Field_II_ver_3_24_linux/gen_samples/'
    label_data_name = ''.join([dataset_name, '.mat'])
    label_size = 32
    dct_size = 25
    label_data = sio.loadmat(''.join([label_data_path,label_data_name]))['phantom_c']
    output_data = np.zeros((num_samples, label_size**2))
    '''
    for i in range(num_samples):
        sam_ind = sam_ls[i]
        print('sam_ind', sam_ind)
        output_data[i,:] = label_data[sam_ind,0][:,:,0].ravel()
        img = label_data[i,0][:,:,0]
# Get DCT Coeff
        dct_coeff = get_2D_dct(img)
        # plt.matshow(np.abs(dct_coeff), cmap=plt.cm.Paired)
# Compress Coeff
        dct_coeff_cp = dct_coeff.copy()
        dct_coeff_cp[dct_size:,:] = 0.0
        dct_coeff_cp[:,dct_size:] = 0.0
        # dct_clip = np.array(filter(lambda x:x>0.0, dct_coeff_cp.reshape(-1,1)))
        dct_clip = dct_coeff_cp[:dct_size,:dct_size].ravel()
    # Alternative
        # v = np.mean(dct_coeff_cp) + 1.0*np.std(dct_coeff_cp)
        # ind = np.nonzero(dct_coeff_cp<v)
        # dct_coeff_cp[ind] = 0.0
        # print("len ind", len(np.array(ind).ravel()))
# Reconstruction
        img_re = get_2D_idct(dct_coeff_cp)
        img_m = np.mean(img_re)
        ind_1 = np.nonzero(img_re>img_m)
        ind_0 = np.nonzero(img_re<img_m)
        img_re[ind_1] = 1
        img_re[ind_0] = 0
        '''

    #with h5.File('output_h5_n2000_82s', 'w') as hf:
    #    hf['output'] = output_data

# Concatenate Files
    with h5.File('input_h5_n1000_119s', 'r') as hf1:
        input1 = np.array(hf1['input'])
        print(input1.shape)
    with h5.File('input_h5_n500_111s', 'r') as hf2:
        input2 = np.array(hf2['input'])
        print(input2.shape)
    with h5.File('input_h5_n100_83s', 'r') as hf3:
        input3 = np.array(hf3['input'])
        print(input3.shape)
    input_train = np.concatenate((input1, input2, input3))
    print(input_train.shape)

    with h5.File('input_h5_n1000_500_100_mixed_313s', 'w') as hf:
        hf['input'] = input_train

    with h5.File('output_h5_n1000_119s', 'r') as hf1:
        output1 = np.array(hf1['output'])
        print(output1.shape)
    with h5.File('output_h5_n500_111s', 'r') as hf2:
        output2 = np.array(hf2['output'])
        print(output2.shape)
    with h5.File('output_h5_n100_83s', 'r') as hf3:
        output3 = np.array(hf3['output'])
        print(output3.shape)
    output_train = np.concatenate((output1, output2, output3))
    print(output_train.shape)

    with h5.File('output_h5_n1000_500_100_mixed_313s', 'w') as hf:
        hf['output'] = output_train


def test_results():
    label_data_path = './'
    label_data_name = 'animals_n100_26-Jul-2016.mat'
    label_data = sio.loadmat(''.join([label_data_path,label_data_name]))['phantom_c']
    print(label_data.shape)

    plt.figure()
    for i in range(10):
        img = label_data[i,0][:,:,0]
        plt.subplot(4,3,i+1)
        plt.imshow(img)

    with h5.File('images_pred_h5', 'r') as hf:
        images = np.array(hf['images_pred'])
    plt.figure()
    for i in range(10):
        img_t = images[i,:,:]
        vm = np.mean(img_t)
        img_t[np.nonzero(img_t<vm)] = 0
        img_t[np.nonzero(img_t>vm)] = 1
        plt.subplot(4,3,i+1)
        plt.imshow(img_t)
    plt.show()


    '''
    l = 0
    m = 0
    n = 0
    for i in range(1,87):
        m = 0
        for j in range(63,129):
            tmp  = sio.loadmat(path.join(prefix, \
                    ''.join(['phant_',str(i),'_rf_ln',str(j),'.mat'])))
            tstart = tmp['tstart']
            rf1 = np.array(tmp['rf_data'][int(tstart*fs):])[0:lenRF,0]
            n = m+1
            for k in range(j+1,129):
                tmp  = sio.loadmat(path.join(prefix, \
                        ''.join(['phant_',str(i),'_rf_ln',str(k),'.mat'])))
                tstart = tmp['tstart']
                rf2 = np.array(tmp['rf_data'][int(tstart*fs):])[0:lenRF,0]

                # Cross Spectrum
                _, csd = signal.csd(rf1, rf2, fs=fs, nperseg=nFFT, \
                        nfft=nFFT, scaling='density')
                input_data[l,0,m,n] = np.abs(np.sum(csd) / nFFT) # sum,average,abs
                n += 1

            input_data[l,0,(m+1):,m] = input_data[l,0,m,(m+1):]
            m += 1
        l += 1

    var_min = np.amin(input_data)
    var_max = np.amax(input_data)
    input_data = 10*((input_data-var_min) / (var_max-var_min))
    print(input_data)

    with h5.File('csm_large_h5', 'w') as hf:
        hf['csm'] = input_data
    '''



if __name__ == '__main__':
    # test_import();
    test_net()
    #  test_results()
