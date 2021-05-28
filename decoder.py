from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Dropout, Lambda, Reshape, Concatenate
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Flatten, Conv1D, Deconvolution2D, Conv2DTranspose
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras import losses
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers.convolutional import UpSampling2D
from tensorflow.keras import initializers
from tensorflow.python.keras.utils.conv_utils import conv_output_length, deconv_length
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import InputSpec
from tensorflow.keras import constraints
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class args:
    epochs = 20
    batch_size = 512
    lr = 0.001
    lr_decay = 0.96
    lam_recon = 0.4
    r = 3
    routings = 3
    shift_fraction = 0.1
    debug = False
    digit = 5
    save_dir = '.'
    t = False
    w = None
    ep_num = 0


decoder = models.Sequential(name='decoder')
decoder.add(Dense(input_dim=32, activation="relu", output_dim=8 * 8 * 64))
decoder.add(Reshape((8, 8, 64)))
decoder.add(BatchNormalization(momentum=0.8))
decoder.add(layers.Deconvolution2D(256, 3, 3, subsample=(1, 1), border_mode='same', activation="relu"))
decoder.add(layers.Deconvolution2D(256, 3, 3, subsample=(2, 2), border_mode='same', activation="relu" ))
decoder.add(layers.Deconvolution2D(256, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
decoder.add(layers.Deconvolution2D(256, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
decoder.add(layers.Deconvolution2D(256, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
# decoder.add(layers.Deconvolution2D(256, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
# decoder.add(layers.Deconvolution2D(8, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
decoder.add(layers.Deconvolution2D(16, 3, 3, subsample=(1, 1), border_mode='same', activation="relu"))
decoder.add(layers.Deconvolution2D(8, 3, 3, subsample=(1, 1), border_mode='same', activation="relu"))
decoder.add(layers.Deconvolution2D(3, 3, 3, subsample=(1, 1), border_mode='same', activation="relu"))
decoder.add(layers.Reshape(target_shape=(128, 128, 3), name='out_recon'))

decoder.summary()

(x_train, y_train), (x_test, y_test) = load_cifar10()

x256_train = resize(x_train)


vect = np.load("decoder_retrain.npy")

lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * 0.5**(epoch // 5))

parallel_model = multi_gpu_model(decoder, gpus=3)

# compile the model  ssim loss for better reconstruction DSSIMObjective()
parallel_model.compile(optimizer=optimizers.Adam(lr=args.lr),
                       loss=['mse'],
                       metrics={'decoder': "loss"})

parallel_model.fit(vect,x256_train, batch_size=32, epochs=args.epochs,callbacks=[lr_decay])
decoder.save(args.save_dir + '/decoder.h5')