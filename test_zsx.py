from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Conv2D, Conv3D, BatchNormalization
from tensorflow.keras.layers import Activation, Dense, Dropout, Lambda, Reshape, Concatenate, Conv3DTranspose
import tensorflow as tf
# from tensorflow.keras.utils import conv_utils
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.conv_utils import conv_output_length
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

from capslayers import Conv2DCaps, Conv3DCaps, ConvCapsuleLayer3D, CapsuleLayer, CapsToScalars, Mask_CID, Mask, ConvertToCaps, FlattenCaps

from globalvariables import *
from myfunctions import get_batches, standardization, checkzeromatrix

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# input_shape = (5, 256, 256, 10, 4)
# x = tf.random.normal(input_shape)

def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))

def MSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''    
    return np.mean((v_ - v) ** 2)

def get_data(hist_len, pred_len):
    data = np.load(data_path_bike+"bike_open.npy")
    data = data[:,:,10:16,:]
    data = np.reshape(np.array(data), (data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))

    data = data[28:28+10, 65:65+10, :]
    
    train_input = data[:, :, 0:300]
    row_total = train_input.shape[2]-hist_len-pred_len
    
    train_x = [train_input[np.newaxis, :, :, x:x+hist_len] for x in range(row_total)]
    train_x = np.concatenate(train_x, axis=0)
    # train_x = train_x[:, :, :, :, np.newaxis]

    train_y = [train_input[np.newaxis, :, :, x+hist_len:x+hist_len+pred_len] for x in range(row_total)]
    train_y = np.concatenate(train_y, axis=0)
    # train_y = train_y[:, :, :, :, np.newaxis]
    
    test_input = data[:, :, -267:]
    row_total = test_input.shape[2]-hist_len-pred_len

    test_x = [test_input[np.newaxis, :, :, x:x+hist_len] for x in range(row_total)]
    test_x = np.concatenate(test_x, axis=0)
    # test_x = test_x[:, :, :, :, np.newaxis]

    test_y = [test_input[np.newaxis, :, :, x+hist_len:x+hist_len+pred_len] for x in range(row_total)]
    test_y = np.concatenate(test_y, axis=0)
    # test_y = test_y[:, :, :, :, np.newaxis]

    return (train_x, train_y), (test_x, test_y)

def model(x_shape, y_shape, routings):
    input_shape = x_shape
    grid_1, grid_2, pred_capsules = y_shape[1:]
    x = Input(shape=input_shape[1:])
    
    # CAPSULE NETWORK
    l = ConvertToCaps()(x)
    l = Conv3D(4, kernel_size=(5,5,5), activation='relu', input_shape=input_shape[1:], padding = 'same')(l)
    l = Conv3D(8, kernel_size=(5,5,5), activation='relu', input_shape=input_shape[1:], padding = 'same')(l) # (5, 254, 254, 8, 32)

    

    l = ConvCapsuleLayer3D(kernel_size=7, num_capsule=pred_capsules, num_atoms=2, strides=1, padding='same', routings=3)(l)

    l = FlattenCaps()(l)

    l = CapsuleLayer(num_capsule=pred_capsules, dim_capsule=32, routings=routings, channels=0, name='digit_caps')(l)

    # # DECODER NETWORK
    # y = Dense(input_dim=pred_capsules, activation="relu", output_dim=grid_1*grid_2)(l)
    y = Dense(grid_1*grid_2*32, activation='relu')(l)
    y = Reshape((grid_1, grid_2, pred_capsules, 32))(y)
    # y = K.permute_dimensions(l,(0,3,1,2,4))
    # y = Reshape((l.shape[3], l.shape[1], l.shape[2], l.shape[4]))(l)
    # print(y.shape)
    # # y = Deconvolution2D(pred_capsules, 3, 3, subsample=(1, 1), border_mode='same')(y)
    # print(K.int_shape(y))
    
    # y = K.permute_dimensions(l,(0,2,3,1,4))
    
    ### DECONVOLUTIONAL LAYERS
    y = Conv3DTranspose(16, (3,3,1), padding = 'same')(y)
    y = Conv3DTranspose(8, (3,3,1), padding = 'same')(y)
    y = Conv3DTranspose(1, (3,3,1), padding = 'same')(y)
    y = Reshape((grid_1, grid_2, pred_capsules))(y)
    # print(y.shape)

    train_model = models.Model(inputs=x, outputs=y)
    train_model.summary()

    return train_model

def train_generator(x, y, batch_size):
    train_datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                        samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.1,
                                        width_shift_range=0.1, height_shift_range=0.1, shear_range=0.0,
                                        zoom_range=0.1, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
                                        vertical_flip=False, rescale=None, preprocessing_function=None,
                                        data_format=None)  # shift up to 2 pixel for MNIST
    train_datagen.fit(x)
    generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=False)
    while True:
        x_batch, y_batch = generator.next()
        #yield ([x_batch, y_batch])    
        yield (x_batch, y_batch)    

def main():
    batch_size = 10
    lr = 0.001
    epochs = 200
    hist_len = 5
    pred_len = 2
    seq_length = hist_len+pred_len

    (train_x, train_y), (test_x, test_y) = get_data(hist_len, pred_len)

    # model(train_x.shape, train_y.shape, 3)
    train_model = model(train_x.shape, train_y.shape, 3)
    

    train_model.compile(optimizer=optimizers.Adam(lr=lr), loss='mean_squared_error', metrics=[metrics.MeanSquaredError()])

    print(train_y.shape)
    y_generator = train_generator(train_x, train_y, batch_size)
    train_model.fit_generator(generator=y_generator, steps_per_epoch=int(train_y.shape[0] / batch_size), epochs=epochs)
    # train_model.fit_generator(generator=train_generator(train_x, train_y, batch_size), steps_per_epoch=int(train_y.shape[0] / batch_size), epochs=epochs, shuffle=False)

    predicted_y = train_model.predict(test_x)

    # # print(type(b1))
    # print(predicted_y.shape)
    # print(test_y.shape)
    mapes = []
    maes = []
    rmses = []
    mses = []
    for i in range(len(predicted_y)):
        y_ = predicted_y[i, :, :, :]
        y = test_y[i, :, :, :]

        mapes.append(MAPE(y, y_))
        maes.append(MAE(y, y_))
        rmses.append(RMSE(y, y_))
        mses.append(MSE(y, y_))
    
    print("=====MAE====")
    print(np.average(np.array(maes)))
    print(np.min(np.array(maes)))
    print(np.max(np.array(maes)))
    print("=====RMSE===")
    print(np.average(np.array(rmses)))
    print(np.min(np.array(rmses)))
    print(np.max(np.array(rmses)))
main()
