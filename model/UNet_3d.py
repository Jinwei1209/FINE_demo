import tensorflow as tf

from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import MaxPooling3D
from keras.layers import UpSampling3D
from keras.layers import Cropping3D
from keras.layers import ZeroPadding3D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Conv3DTranspose
from keras.layers import Concatenate
from keras.models import Model
from keras.layers.core import Lambda

class MirrorPadding3D(ZeroPadding3D):
    '''
    Pad input in a reflective manner
    '''
    def call(self, x):
        pattern = [[0, 0],
                   list(self.padding[0]),
                   list(self.padding[1]),
                   list(self.padding[2]),
                   [0, 0]]
        return tf.pad(x, pattern, mode='SYMMETRIC')

def conv_act_block(x, 
                   filter_size, kernel_size, 
                   use_bn=True, 
                   name_template=''):
    padding = tuple( x//2 for x in kernel_size )
    x = Conv3D(filter_size, kernel_size=kernel_size, kernel_initializer='he_normal', name=name_template.format('conv1'))(MirrorPadding3D(padding)(x))
    x = Activation('relu', name=name_template.format('act1'))(x)
    if use_bn:
        x = BatchNormalization(name=name_template.format('bn1'))(x)
    x = Conv3D(filter_size, kernel_size=kernel_size, kernel_initializer='he_normal', name=name_template.format('conv2'))(MirrorPadding3D(padding)(x))
    x = Activation('relu', name=name_template.format('act2'))(x)
    if use_bn:
        x = BatchNormalization(name=name_template.format('bn2'))(x)
    return x

def UNet_3d(img_size, filter_base=32, kernel_size=(3,3,3),
            level=5, use_bn=True, use_deconv=True):
    inputs = Input(img_size + (1,), name='input')
    x = inputs
    
    filter_sizes = [ min(512, filter_base * 2**i) for i in range(level) ]
    endpoints = []

    # down-stream
    for i in range(level-1):
        filter_size = filter_sizes[i]
        name_template = '{{}}_{0}d'.format(i+1)

        # Conv-Act-BN
        x = conv_act_block(x, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)
        endpoints.append(x)

        # Down-sampling
        x = MaxPooling3D(pool_size=(2,2,2), padding='same', name=name_template.format('down'))(x)

    # last level
    filter_size = filter_sizes[level-1]
    name_template = '{{}}_{0}'.format(level)
    # Conv-Act-BN
    x = conv_act_block(x, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)
    
    # up-stream
    for i in range(level-2, -1, -1):
        filter_size = filter_sizes[i]
        name_template = '{{}}_{0}u'.format(i+1)

        # Up-sampling
        if use_deconv:
            x = Conv3DTranspose(filter_size, kernel_size, strides=2, padding='valid', kernel_initializer='he_normal', name=name_template.format('deconv'))(x)
            x = Activation('relu', name=name_template.format('actdc'))(x)
            if use_bn:
                x = BatchNormalization(name=name_template.format('bndc'))(x)            
        else:
            x = UpSampling3D(size=(2,2,2), name=name_template.format('up'))(x)
        # Concatenation (crop if needed)
        y = endpoints[i]
        shape_x1, shape_y = x._keras_shape[1:-1], y._keras_shape[1:-1]
        cropping = tuple( ((i-j)//2, (i-j+1)//2) for i,j in zip(shape_x1, shape_y) )
        x = Cropping3D(cropping=cropping)(x)
        x = Concatenate(axis=-1, name=name_template.format('cat'))([x, y])

        # Conv-Act-BN
        x = conv_act_block(x, filter_size=filter_size, kernel_size=kernel_size, use_bn=use_bn, name_template=name_template)

    outputs = Conv3D(1, kernel_size=(1,1,1), kernel_initializer='he_normal', name='conv_final')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model