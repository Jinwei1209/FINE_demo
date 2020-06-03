import os
import glob
import tensorflow as tf
import numpy as np
import scipy.io as sio

from keras.models import Model, Sequential
from keras import metrics
from keras.optimizers import *
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from model.UNet_3d import UNet_3d
from utils.utils import *

# Configure TensorFlow session (memory allocation)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# parameters
data_load_dir = './data'
data_save_dir = './result/cosmos/'
i_case = 'COSMOS_test'
factor = 3.693
voxel_size = [0.9375, 0.9375, 3]
B0_dir = [0,0,1]
lambda_bg = 1e+02
epochs = 30

# load data
# ground truth
filename = '{0}/{1}/COSMOS_smv_3mm.mat'.format(data_load_dir, i_case)
QSMs = np.real(load_mat(filename, varname='COSMOS_new'))
# data used for FINE step
filename = '{0}/{1}/Mask_smv_3mm.mat'.format(data_load_dir, i_case)
Masks = np.real(load_mat(filename, varname='Mask_new'))
filename = '{0}/{1}/RDF_smv_3mm.mat'.format(data_load_dir, i_case)
RDFs = np.real(load_mat(filename, varname='RDF_new'))
filename = '{0}/{1}/N_std_smv_3mm.mat'.format(data_load_dir, i_case)
N_stds = np.real(load_mat(filename, varname='N_std_new'))

# iterate over orientations
for i_dir in range(5):
    Mask = Masks[..., i_dir]
    QSM = QSMs[..., i_dir] * Mask
    RDF_input = RDFs[..., i_dir] * Mask
    N_std = N_stds[..., i_dir]

    vol_size = Mask.shape
    D = dipole_kernel(vol_size, voxel_size, B0_dir)

    tempn = np.double(N_std*Mask)
    Data_weights = np.real(dataterm_mask(tempn, Mask))
    noise = np.random.randn(vol_size[0], vol_size[1], vol_size[2]) * N_std
    RDF_in_loss = np.fft.ifftn(np.fft.fftn(QSM)*D)*Mask + noise*Mask
    RDF_input = RDF_in_loss

    # Loss
    D = tf.convert_to_tensor(D, np.complex64)
    def fidelity_loss(y_true, y_pred):
        
        weights = tf.convert_to_tensor(Data_weights, tf.float32)
        y_pred_cplx = tf.cast(y_pred[0, ..., 0], tf.complex64)
        y_pred_RDF = tf.cast(tf.real(tf.ifft3d(tf.fft3d(y_pred_cplx)*D)), tf.complex64)
        measured_RDF = tf.cast(y_true[0, ..., 0], tf.complex64)
        diff = tf.abs(tf.exp(1j*y_pred_RDF*factor) - tf.exp(1j*measured_RDF*factor)) # nonlinaer fidelity
    #     diff = tf.abs(y_pred_RDF - measured_RDF) # linear fidelity 
        loss = 1000*tf.reduce_mean(tf.square(weights*diff))
        return loss

    def background_loss(y_true, y_pred):
        backgroud_mask = tf.convert_to_tensor(1-Mask[np.newaxis,...,np.newaxis], tf.float32)
        loss = lambda_bg*tf.reduce_mean(tf.square(backgroud_mask*y_pred))
        return loss

    def loss_total(y_true, y_pred):
        return fidelity_loss(y_true, y_pred)

    # QSMnet
    model = UNet_3d(vol_size, use_bn=False, use_deconv=True, filter_base=32)
    model.load_weights('./weight/pre_trained_weight.h5')
    QSMnet = model.predict(RDF_input[np.newaxis,...,np.newaxis], batch_size=1, verbose=1)
    print('Successfully get QSMnet output')

    adict = {}
    adict['QSMnet'] = QSMnet[0,:,:,:,0]*Mask
    sio.savemat(data_save_dir+'QSMnet_dir{0}.mat'.format(i_dir+1), adict)

    adict = {}
    adict['COSMOS'] = QSM*Mask
    sio.savemat(data_save_dir+'COSMOS_dir{0}.mat'.format(i_dir+1), adict)

    # FINE Step
    RDF_input = np.repeat(RDF_input[np.newaxis,...,np.newaxis], 2, axis=0)
    RDF_in_loss = np.repeat(RDF_in_loss[np.newaxis,...,np.newaxis], 2, axis=0)
    model.compile(loss = loss_total, optimizer = Adam(lr=1e-4), metrics=[fidelity_loss, background_loss])
    model.fit(RDF_input, RDF_in_loss, epochs=epochs, batch_size=1, shuffle=False, validation_split=0.5)
    pred = model.predict(RDF_input[0:1, ...], batch_size=1, verbose=1)
    pred_QSM = pred[0, ..., 0]
    print('Successfully get FINE output')

    # save data
    adict = {}
    adict['QSM_refine'] = pred_QSM
    sio.savemat(data_save_dir+'QSM_refine_dir{0}.mat'.format(i_dir+1), adict)
    print('Successfully save data')




