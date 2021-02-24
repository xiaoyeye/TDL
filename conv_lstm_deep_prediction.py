"""
#This script demonstrates the use of a convolutional LSTM network.

This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten,MaxPooling3D
import numpy as np
import pylab as plt
from keras.optimizers import SGD
from keras.utils import plot_model
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
## A new gene pair list with 3D NEPDF is fed to the trained model and the result is prediction labels for the new list.
## The separation file here has only two numbers, 0 in the 1st line, and the total number in the 2nd line.

def load_data(indel_list,data_path): # cell type specific  
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = np.load(data_path+'/NTxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        for k in range(len(ydata)):
            xxdata_list.append(xdata[k,:,:,:,:])
            yydata.append(ydata[k])
        count_setx = count_setx + len(ydata)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))


data_augmentation = False
batch_size = 1024
num_classes = 2
epochs = 100
model_name = 'keras_cnn_trained_model_shallow.h5'
data_path = '/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/NT_8X8_6' ###### A completely new gene pair list that the model will predict
###################################################################################################################################
for test_indel in [1]: ################## 
    (x_test, y_test, count_set) = load_data([0], data_path)
    #(x_test, y_test, count_set) = load_data(test_TF, data_path)
    print(x_test.shape, 'x_test samples')
    print(y_test.shape, 'y_test samples')
############################### model
    save_dir = os.path.join(os.getcwd(),str(test_indel) + 'new_Xlr00005_KEGG_3d_conv_ddeeper_NT_p600_e' + str(epochs)) ##### The path of model used here
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),input_shape=x_test.shape[1:],padding='valid', return_sequences=True,dropout=0.5,recurrent_dropout=0.5))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),padding='valid', return_sequences=True,dropout=0.5,recurrent_dropout=0.5))
    seq.add(BatchNormalization())
    seq.add(MaxPooling3D(pool_size=(2,2,2),padding='same',data_format='channels_last'))
    # seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),padding='valid', return_sequences=True,dropout=0.5,recurrent_dropout=0.5))
    # seq.add(BatchNormalization())
    seq.add(Flatten())
    seq.add(Dense(512))
    seq.add(Activation('relu'))
    seq.add(Dropout(0.5))
    seq.add(Dense(1, activation='sigmoid'))
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    seq.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
    plot_model(seq, to_file='conv_lstm.png',show_shapes=True)
    model_path = os.path.join(save_dir, model_name)
    seq.load_weights(model_path) ############ load the model parameters
    print('load model and predict')
    y_predict = seq.predict(x_test)
    np.save(save_dir + '/end_y_predict.npy', y_predict)     ###### prediction result
