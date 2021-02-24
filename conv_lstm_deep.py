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
# We create a layer which take as input movies of shape
# 
# 

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


data_augmentation = False
batch_size = 1024
num_classes = 2
epochs = 100
length_TF = 36 ## the number of TFs
model_name = 'keras_cnn_trained_model_shallow.h5'
whole_data_TF = [i for i in range(length_TF)]
data_path = '/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/GTRD_NT_8X8_6'  ##the path of generated 3D NEPDF files and their ground truth
###################################################################################################################################
for test_indel in range(1,4): ################## three fold cross validation
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*length_TF)),int(np.ceil(test_indel*0.333333*length_TF)))]
    #test_TF = [test_indel]
    train_TF = [i for i in whole_data_TF if i not in test_TF]                                                                  #
    # indel_train_list = [0,1]
    # indel_test_list = [2]
    (x_train, y_train, count_set_train) = load_data(train_TF, data_path)
    (x_test, y_test, count_set) = load_data(test_TF, data_path)
    print(x_train.shape, 'x_train samples')
    print(x_test.shape, 'x_test samples')
    print(y_train.shape, 'y_train samples')
    print(y_test.shape, 'y_test samples')
############################### model
    save_dir = os.path.join(os.getcwd(),str(test_indel) + 'new_Xlr00001_KEGG_3d_conv_ddeeper_NT_p600_e' + str(epochs))
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
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    seq.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
    plot_model(seq, to_file='conv_lstm.png',show_shapes=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=600, verbose=0, mode='auto') ## user can set patience as 50, the half of max epoch number.
    ### we set it as 600 since after several trials, we found that  the model trained for 100 epochs can get pretty good results.So the monitoring for validation accuracy 
    ### was used to find the best model during the 100 epoch train. And we compared the model at the end of training and the model of max validation accuracy, and it was found
    ### that model at end always has better result.
    checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                  verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_acc', verbose=1,
                                  save_best_only=True, mode='auto', period=1)
    callbacks_list = [checkpoint2, early_stopping]
    if not data_augmentation:
        print('Not using data augmentation.')
        history = seq.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,validation_split=0.2,
                  shuffle=True, callbacks=callbacks_list)

    # Save model and weights
    model_path = os.path.join(save_dir, model_name)
    seq.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    scores = seq.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    y_predict = seq.predict(x_test)
    np.save(save_dir+'/end_y_test.npy',y_test)
    np.save(save_dir+'/end_y_predict.npy',y_predict)
    ############################################################################## plot training process
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.savefig(save_dir + '/end_result.pdf')
    ###############################################################  evaluation without consideration of data separation
    plt.figure(figsize=(10, 6))
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    plt.plot(fpr, tpr)
    plt.grid()
    plt.plot([0, 1], [0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    auc = np.trapz(tpr, fpr)
    print('AUC:', auc)
    plt.savefig(save_dir + '/overall.pdf')
    #######################################

    #############################################################
        #########################
    y_testy = y_test
    y_predicty = y_predict
    fig = plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1])
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('FP')
    plt.ylabel('TP')
    # plt.grid()
    AUC_set = []
    s = open(save_dir + '/divided_interaction.txt', 'w')
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)  # 3068
    for jj in range(len(count_set) - 1):  # len(count_set)-1):
        if count_set[jj] < count_set[jj + 1]:
            print(test_indel, jj, count_set[jj], count_set[jj + 1])
            y_test = y_testy[count_set[jj]:count_set[jj + 1]]
            y_predict = y_predicty[count_set[jj]:count_set[jj + 1]]
            # Score trained model.
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # Print ROC curve
            plt.plot(fpr, tpr, color='0.5', lw=0.001, alpha=.2)
            auc = np.trapz(tpr, fpr)
            s.write(str(jj) + '\t' + str(count_set[jj]) + '\t' + str(count_set[jj + 1]) + '\t' + str(auc) + '\n')
            print('AUC:', auc)
            AUC_set.append(auc)

    mean_tpr = np.median(tprs, axis=0)
    mean_tpr[-1] = 1.0
    per_tpr = np.percentile(tprs, [25, 50, 75], axis=0)
    mean_auc = np.trapz(mean_tpr, mean_fpr)
    plt.plot(mean_fpr, mean_tpr, 'k', lw=3, label='median ROC')
    plt.title(str(mean_auc))
    plt.fill_between(mean_fpr, per_tpr[0, :], per_tpr[2, :], color='g', alpha=.2, label='Quartile')
    plt.plot(mean_fpr, per_tpr[0, :], 'g', lw=3, alpha=.2)
    plt.legend(loc='lower right')
    plt.savefig(save_dir + '/divided_interaction_percentile.pdf')
    del fig
    fig = plt.figure(figsize=(5, 5))
    plt.hist(AUC_set, bins=50)
    plt.savefig(save_dir + '/divided_interaction_hist.pdf')
    del fig
    s.close()