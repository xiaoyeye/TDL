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
import os,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
## A new gene pair list with 3D NEPDF is fed to the trained model and the result is prediction labels for the new list.
## The separation file here has only two numbers, 0 in the 1st line, and the total number in the 2nd line.

def load_data(indel_list,data_path): # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:#len(h_tf_sc)):
        xdata = np.load(data_path+'/NTxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path+'/ydata_tf' + str(i) + '.npy')
        for k in range(int(len(ydata)/3)):
            xxdata_list.append(xdata[3*k,:,:,:,:]) ## actually the TF-candidate list we provide has three labels, 1 for TF->target, 2 for target->TF, 0 for TF->non target
            xxdata_list.append(xdata[3*k+2,:,:,:,:])  ## label 1 0 are selected for interaction task; label 1 2 are selected for causality task.
            yydata.append(1)
            yydata.append(0)
        count_setx = count_setx + int(len(ydata)*2/3)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))


	
	
# Script starts from here
if len(sys.argv) < 3:
    print ('No enough input files')
    sys.exit()

model_path = sys.argv[1]
data_path = sys.argv[2]
model_name = 'keras_cnn_trained_model_shallow.h5'
#model_path = 
#data_path = '/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/GTRD_NT_8X8_6' ###### A completely new gene pair list that the model will predict
### In the demo, for simplification, we used the same list that used to train the model. Users can provide their own 3D NEPDF generated from new gene pair list.
###################################################################################################################################

(x_test, y_test, count_set) = load_data([0], data_path)
#(x_test, y_test, count_set) = load_data(test_TF, data_path)
print(x_test.shape, 'x_test samples')
print(y_test.shape, 'y_test samples')
############################### model
# save_dir = os.path.join(os.getcwd(),str(test_indel) + 'new_Xlr00005_KEGG_3d_conv_ddeeper_NT_p600_e' + str(epochs)) ##### The path of model used here
# if not os.path.isdir(save_dir):
	# os.makedirs(save_dir)
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
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
seq.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
plot_model(seq, to_file='conv_lstm.png',show_shapes=True)
#model_path = os.path.join(save_dir, model_name)
seq.load_weights(model_path+'/'+model_name) ############ load the model parameters
print('load model and predict')
y_predict = seq.predict(x_test)
np.save(model_path + '/newdata_y_predict.npy', y_predict)     ###### prediction result
