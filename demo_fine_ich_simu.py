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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# parameters
data_load_dir = './data'
data_save_dir = './result/ich_simu/'
i_case = 'ICH_simulation14'
factor = 3.693
voxel_size = [1, 1, 3]
B0_dir = [0,0,1]
lambda_bg = 1e+02
epochs = 50

# load data
# ground truth
filename = '{0}/{1}/qsm_simu.mat'.format(data_load_dir, i_case)
QSM = np.real(load_mat(filename, varname='qsm_simu'))

# data used for FINE step
filename = '{0}/{1}/rdf_simu.mat'.format(data_load_dir, i_case)
RDF = np.real(load_mat(filename, varname='rdf_simu'))
filename = '{0}/{1}/N_std_simu.mat'.format(data_load_dir, i_case)
N_std = np.real(load_mat(filename, varname='N_std_simu')) 

Mask = abs(QSM) > 0
QSM = QSM*Mask
RDF_input= RDF*Mask
RDF_in_loss = RDF

vol_size = Mask.shape[:3]
D = dipole_kernel(vol_size, voxel_size, B0_dir)

tempn = np.double(N_std*Mask)
Data_weights = np.real(dataterm_mask(tempn, Mask))

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
sio.savemat(data_save_dir+'QSMnet.mat',adict)

adict = {}
adict['label'] = QSM*Mask
sio.savemat(data_save_dir+'label.mat',adict)

# FINE step
RDF_input = np.repeat(RDF_input[np.newaxis,...,np.newaxis], 2, axis=0)
RDF_in_loss = np.repeat(RDF_in_loss[np.newaxis,...,np.newaxis], 2, axis=0)
model.compile(loss=loss_total, optimizer=Adam(lr=1e-4), metrics=[fidelity_loss, background_loss])

for i in range(1, 4): 
    model.fit(RDF_input, RDF_in_loss, epochs=epochs, batch_size=1, shuffle=False, validation_split=0.5)
    pred = model.predict(RDF_input[0:1, ...], batch_size=1, verbose=1)
    pred_QSM = pred[0, ..., 0]
    print('Successfully get FINE output')

    # save data
    adict = {}
    adict['QSM_refine'] = pred_QSM
    sio.savemat(data_save_dir+'QSM_refine_{0}.mat'.format(epochs*i), adict)
    print('Successfully save data')




