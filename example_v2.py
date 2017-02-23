#!/usr/bin/env python
# encoding: utf-8

'''
Example 1 for acoular library.
'''

import acoular
from acoular import L_p, Calib, MicGeom, EigSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional
from numpy import zeros
from os import path
from pylab import figure, subplot, imshow, show, colorbar, title

dataFile = 'example_data.h5'
calibFile = 'example_calib.xml'
micGeoFile = path.join(path.split(acoular.__file__)[0], 'xml', 'array_56.xml')

freqInt = 4000

t1 = MaskedTimeSamples(name = dataFile)
t1.start = 0
t1.stop = 16000
invalid = [1, 7]
t1.invalid_channels = invalid

t1.calib = Calib(from_file = calibFile)

m = MicGeom(from_file = micGeoFile)
m.invalid_channels = invalid

g = RectGrid(x_min = -0.6, x_max = -0.0, y_min = -0.3, y_max = 0.3,
        z = 0.68, increment = 0.05)

f = EigSpectra(time_data = t1,
        window = 'Hanning', overlap = '50%', block_size = 128,
        ind_low = 7, ind_high = 15)

bb = BeamformerBase(freq_data = f, grid = g, mpos = m, r_diag = True, c = 346.04)
bc = BeamformerCapon(freq_data = f, grid = g, mpos = m, c = 346.04, cached = False)
be = BeamformerEig(freq_data = f, grid = g, mpos = m, r_diag = True, c = 346.04, n = 54)
bm = BeamformerMusic(freq_data = f, grid = g, mpos = m, c = 346.04, n = 6)

bd = BeamformerDamas(beamformer = bb, n_iter = 100)
bo = BeamformerOrth(beamformer = be, eva_list = range(38, 54))

bs = BeamformerCleansc(freq_data = f, grid = g, mpos = m, r_diag = True, c = 346.04)
bf = BeamformerCMF(freq_data = f, grid = g, mpos = m, c = 346.04, method = 'LassoLarsBIC')
bl = BeamformerClean(beamformer = bb, n_iter = 100)
bn = BeamformerFunctional(freq_data = f, grid = g, mpos = m, r_diag = True, c = 346.04, gamma = 4)

figure(1)
i1 = 1
for b in (bb, bc, be, bm, bd, bo, bs, bf, bl, bn):
    subplot(3, 4, i1)
    i1 += 1
    map = b.synthetic(freqInt, 1)
    mx = L_p(map.max())
    imshow(L_p(map.T), vmax = mx, vmin = mx - 15, interpolation = 'nearest', extent = g.extend())
    colorbar()
    title(b.__class__.__name__)

bt = BeamformerTime(source = t1, grid = g, mpos = m, c = 346.04)
ft = FiltFiltOctave(source = bt, band = freqInt)
pt = TimePower(source = ft)
avgt = TimeAverage(source = pt, naverage = 1024)
cacht = TimeCache(source = avgt)

fi = FiltFiltOctave(source = t1, band = freqInt)
bts = BeamformerTimeSq(source = fi, grid = g, mpos = m, r_diag = True, c = 346.04)
avgts = TimeAverage(source = bts, naverage = 1024)
cachts = TimeCache(source = avgts)

i2 = 2
for b in (cacht, cachts):
    figure(i2)
    i2 += 1
    res = zeros(g.size)
    i3 = 1
    for r in b.result(1):
        subplot(4, 4, i3)
        i3 += 1
        res += r[0]
        map = r[0].reshape(g.shape)
        mx = L_p(map.max())
        imshow(L_p(map.T), vmax = mx, vmin = mx - 15, interpolation = 'nearest', extent = g.extend())
        title('%i' % ((i3 - 1) * 1024))
    res /= i3 - 1
    figure(1)
    subplot(3, 4, i1)
    i1 += 1
    map = r[0].reshape(g.shape)
    mx = L_p(map.max())
    imshow(L_p(map.T), vmax = mx, vmin = mx - 15, interpolation = 'nearest', extent = g.extend())
    colorbar()
    title(('BeamformerTime', 'BeamformerTimeSq')[i2 - 3])
show()
