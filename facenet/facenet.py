from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K

from .layers import *

from tensorflow.keras.optimizers import Adagrad
from tensorflow_addons.losses import TripletSemiHardLoss

from functools import partial

conv = partial(Conv2D, padding='same', activation='relu')
dense = partial(Dense, activation='linear')

max_pooling = partial(MaxPooling2D, padding='same')
l2_pooling = partial(L2Pooling, padding='same')

def inception_module(input_tensor, filters, pooling='max', name='inception'):
    f1, f2, f3, f4, f5, f6 = filters
    
    tower1 = conv(f1, (1, 1), 1, name=name + '/1x1')(input_tensor)
    
    tower2 = conv(f2, (1, 1), 1, name=name + '/3x3_reduce')(input_tensor)
    tower2 = conv(f3, (3, 3), 1, name=name + '/3x3')(tower2)
    
    tower3 = conv(f4, (1, 1), 1, name=name + '/5x5_reduce')(input_tensor)
    tower3 = conv(f5, (5, 5), 1, name=name + '/5x5')(tower3)
    
    if pooling == 'max':
        tower4 = max_pooling((3, 3), 1, name=name + '/pool')(input_tensor)
    elif pooling == 'l2':
        tower4 = l2_pooling((3, 3), 1, name=name + '/pool')(input_tensor)
    tower4 = conv(f6, (1, 1), 1, name=name + '/pool_proj')(tower4)
    
    towers = [tower1, tower2, tower3, tower4]
    conc = Concatenate(name=name + '/conc')(towers)
    
    return conc

def inception_module_partial(input_tensor, filters, strides=(1, 2, 1, 2), name='inception'):
    f2, f3, f4, f5 = filters
    s2, s3, s4, s5 = strides
    
    tower2 = conv(f2, (1, 1), s2, name=name + '/3x3_reduce')(input_tensor)
    tower2 = conv(f3, (3, 3), s3, name=name + '/3x3')(tower2)
    
    tower3 = conv(f4, (1, 1), s4, name=name + '/5x5_reduce')(input_tensor)
    tower3 = conv(f5, (5, 5), s5, name=name + '/5x5')(tower3)
    
    tower4 = max_pooling((3, 3), 2, name=name + '/pool')(input_tensor)
    
    towers = [tower2, tower3, tower4]
    conc = Concatenate(name=name + '/conc')(towers)
    
    return conc

def create_facenet_nn2(image_size, channels, alpha, lr):
    inp = Input((*image_size, channels), name='input')

    out = conv(64, (7, 7), 2, name='conv1')(inp)
    out = max_pooling((3, 3), 2, name='pool1')(out)
    out = LRN(name='lrn1')(out)

    out = conv(64, (1, 1), 1, name='inception2' + '/3x3_reduce')(out)
    out = conv(192, (3, 3), 1, name='inception2' + '/3x3')(out)

    out = LRN(name='lrn2')(out)
    out = max_pooling((3, 3), 2, name='pool2')(out)

    out = inception_module(out, (64, 96, 128, 16, 32, 32), pooling='max', name='inception_3a')
    out = inception_module(out, (64, 96, 128, 32, 64, 64), pooling='l2', name='inception_3b')
    out = inception_module_partial(out, (128, 256, 32, 64), name='inception_3c')

    out = inception_module(out, (256, 96, 192, 32, 64, 128), pooling='l2', name='inception_4a')
    out = inception_module(out, (224, 112, 224, 32, 64, 128), pooling='l2', name='inception_4b')
    out = inception_module(out, (192, 128, 256, 32, 64, 128), pooling='l2', name='inception_4c')
    out = inception_module(out, (160, 144, 288, 32, 64, 128), pooling='l2', name='inception_4d')
    out = inception_module_partial(out, (160, 256, 64, 128), name='inception_4e')

    out = inception_module(out, (384, 192, 384, 48, 128, 128), pooling='max', name='inception_5a')
    out = inception_module(out, (384, 192, 384, 48, 128, 128), pooling='l2', name='inception_5b')

    out = GlobalAveragePooling2D(name='avg_pool')(out)
    out = Dropout(0.4)(out)
    out = dense(128, name='fc')(out)
    out = L2Normalize(name='embeddings')(out)

    facenet = Model(inp, out, name='FaceNet_NN2')
    facenet.summary()

    triplet_loss = TripletSemiHardLoss(alpha)
    sgd_opt = Adagrad(lr)
    facenet.compile(sgd_opt, triplet_loss)

    return facenet
