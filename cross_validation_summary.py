from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import interp
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")
data_augmentation = False
# num_predictions = 20
batch_size = 256
num_classes = 3
epochs = 200
data_augmentation = False
# num_predictions = 20
model_name = 'keras_cnn_trained_model_shallow.h5'
# The data, shuffled and split between train and test sets:


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
            xxdata_list.append(xdata[3*k,:,:,:,:])
            xxdata_list.append(xdata[3*k+2,:,:,:,:])
            yydata.append(1)
            yydata.append(0)
        count_setx = count_setx + int(len(ydata)*2/3)
        count_set.append(count_setx)
        print (i,len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print (np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))



save_dir = os.path.join(os.getcwd(),'new_Xlr00001_KEGG_3d_conv_ddeeper_NT_p600_e100_threesets_acc') ### the final performance folder
length_TF = 36
model_name = 'keras_cnn_trained_model_shallow.h5'
whole_data_TF = [i for i in range(length_TF)]
data_path = '/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/GTRD_NT_8X8_6' ### 3D NEPDF folder
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    # plt.grid()
AUC_set =[]
s = open(save_dir+'/whole_RPKM_AUCs1+2.txt','w')
tprs = []
mean_fpr = np.linspace(0, 1, 100)

y_testy = np.empty([0])
y_predicty = np.empty([0,1])
#count_setx = pd.read_table('/home/yey3/nn_project2/data/human_brain/pathways/kegg/unique_rand_labelx_num.txt',header=None)
#count_set = [i[0] for i in np.array(count_setx)]
count_set = [0]
for test_indel in range(1,4):
    test_TF = [i for i in range (int(np.ceil((test_indel-1)*0.333333*length_TF)),int(np.ceil(test_indel*0.333333*length_TF)))]
    print (len(test_TF))
    (x_testx, y_testx, count_setz) = load_data(test_TF, data_path)
    y_predictyz = np.load('/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/program2/seperation/conv_lstm/'+str(test_indel)+'new_Xlr00005_KEGG_3d_conv_ddeeper_NT_p600_e100' + '/end_y_predict.npy') ### trained model for each fold cross validation
    y_testyz = np.load('/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/program2/seperation/conv_lstm/'+str(test_indel)+'new_Xlr00005_KEGG_3d_conv_ddeeper_NT_p600_e100' + '/end_y_test.npy')	### trained model for each fold cross validation
    y_testy = np.concatenate((y_testy,y_testyz),axis = 0)
    y_predicty = np.concatenate((y_predicty, y_predictyz), axis=0)
    count_set = count_set + [i + count_set[-1] if len(count_set)>0 else i for i in count_setz[1:]]
    ############
print (len(count_set))
###############whole performance

##################################
fig = plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1])
total_pair = 0
total_auc = 0
print (y_predicty.shape)
    ############
for jj in range(len(count_set)-1):#len(count_set)-1):
    if count_set[jj] < count_set[jj+1]:
        print (test_indel,jj,count_set[jj],count_set[jj+1])
        current_pair = count_set[jj+1] - count_set[jj]
        total_pair = total_pair + current_pair
        y_test = y_testy[count_set[jj]:count_set[jj+1]]
        y_predict = y_predicty[count_set[jj]:count_set[jj+1]]
        # Score trained model.
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)		
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
            # Print ROC curve
        plt.plot(fpr, tpr, color='0.5', lw=0.001,alpha=.2)
        auc = np.trapz(tpr, fpr)
        s.write(str(jj)+'\t'+str(count_set[jj])+'\t'+str(count_set[jj+1])+'\t'+str(auc) + '\n')
        print('AUC:', auc)
        AUC_set.append(auc)
        total_auc = total_auc + auc * current_pair

mean_tpr = np.median(tprs, axis=0)
mean_tpr[-1] = 1.0
per_tpr = np.percentile(tprs,[25,50,75],axis=0)
mean_auc = np.trapz(mean_tpr,mean_fpr)
plt.plot(mean_fpr, mean_tpr,'k',lw=3,label = 'median ROC')
plt.title("{:.4f}".format(mean_auc),fontsize=15)
plt.fill_between(mean_fpr, per_tpr[0,:], per_tpr[2,:], color='g', alpha=.2,label='quantile')
plt.plot(mean_fpr, per_tpr[0,:],'g',lw=3,alpha=.2)
plt.legend(loc='lower right',fontsize=15)
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.grid()
plt.xlabel('FP', fontsize=15)
plt.ylabel('TP', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(save_dir+'/whole_kegg_ROCs1+2_percentile.pdf')
del fig
fig = plt.figure(figsize=(3, 3))
plt.hist(AUC_set,bins = 50)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_hist.pdf')
del fig
s.close()
fig = plt.figure(figsize=(3, 3))
plt.boxplot(AUC_set)
plt.savefig(save_dir + '/whole_kegg_ROCs1+2_box.pdf')
del fig
############################


