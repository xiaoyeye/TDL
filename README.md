# Time-course Deep Learning (TDL) models for gene interaction inference
# Title, Deep learning of gene interactions from single cell time-course expression data
##

># 1, TDL overview
![](https://github.com/xiaoyeye/TDL/blob/master/TDL_overview.png)

Time-course deep learning (TDL) model architecture. To infer gene interactions (top left) we first convert time-course single cell expression data to a 3D tensor, which we term normalized empirical probability distribution function (NEPDF). Each 2D slice of the NEPDF captures the co-expression of a pair of genes at one of the time points profiled and the 3D NEPDF represents their co-expression over time. 3D NEPDF is then used as input to atime-course deep learning (TDL) model. The model is trained using labeled positive and negative pairs. The figure shows the convolutional Long short-term memory (LSTM) architecture which is one of the two TDL model we tested. This mode consists of two stacked LSTM layers, followed by a dense layer which concatenates all convolutional hidden state from LSTM layer and then a final output (classifica-tion) layer. See Supporting Figure S1 for the other TDL architecture we tested, 3D CNN.

># 2, Code environment

>>## Users need to install python and ‘Keras’ and ‘theano’ modules, and  all ohther modules required by the code. We  recommend Anaconda to do this.
Author's environment is python 3.6.3 in a Linux server which is now running Centos 6.5 as the underlying OS and Rocks 6.1.1 as the cluster management revision. 

># 3, Demo for running in interaction task  
![](https://github.com/xiaoyeye/TDL/blob/master/pipeline_for_TDL.jpg)

Firstly, gene pair plus label list and normalized time-course single cell gene expression data are provided by users to generated 3D NEDPF for each (gene pair, label). Secondly, a cross validation strategy is used to train and test our TDL models. Finally the model can predict new gene pair’s label.

## Users should first figure out the right paths for all the scripts and datasets.
>>## 3.0 gene pair list generation from known gene pair set 
>>>### Usage: 

    python gene_pair_set_generation.py <gene name reference list> <known gene pair list> <path of generated pair list> <path of generated pair num list>
    
>>>### command line in author's linux machine :

    python gene_pair_set_generation.py /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/hesc2_gene_list_ref.txt /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/hesc2_TF_target.txt /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/hesc2_gene_pairs_400.txt /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/hesc2_gene_pairs_400_num.txt

`gene_pair_set_generation.py` uses the konwn gene pair list to generate a gene pair list where konwn gene pair is positve sample and random pair is negative sample.

>>>#1, `gene name reference list` is the list that converts sc expression data gene set into gene symbol IDs. Format: `'gene symbol IDs\t sc gene ID'`.

>>>#2, `known pair list` is the list that contains konwn gene pairs. format : `'GeneA    GeneB'`.

>>>#3, `path of generated pair list` is the destination of the list that contains positive and negative gene pairs and their labels. format : `'GeneA    GeneB    label'`.

>>>#4, `path of generated pair num list` is the destination of a number list that divides gene_pair_list into small parts, where each part corresponds to one TF.


#################OUTPUT

It generates a gene pair list with positive and negative gene pairs, and also a gene pair number list to divide gene_pair_list into small parts, where each part corresponds to one TF.

Here we use gene symbol information to align scRNA-seq and gene pair's gene sets. The scRNA-seq may use entrez ID or ensembl ID, or other IDs, gene pair list for GTRD used gene symbol ID, thus we generated `gene name reference list` to convert all the IDs to gene symbols. Please also do IDs convertion if users want to use their own expression data.

## The file path of 'data_path'in the following code is used in the author's own machine, so please change it to the approprite path according to user's own environment.


>>## 3.1 Training and test data generation 
>>>### Usage: 

    python 3D_tensor_generation.py <None> <gene name reference list> <TF-target pair list> <TF-target pair count list> <None> <time-course scRNA-seq path+file prefix> <time point number> <flag if pair list has ground truth> <data generation folder name> 
    
>>>### command line in author's linux machine :

    python 3D_tensor_generation.py None /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/hesc2_gene_list_ref.txt /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/hesc2_gene_pairs_400.txt /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/hesc2_gene_pairs_400_num.txt None /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/RPKM 6 1 GTRD_NT_8X8_6 
 (where the path for the expession data is 
 
 '/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/RPKM_0.h5',
 
 '/home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/RPKM_1.h5'......)

`3D_tensor_generation.py` uses the normalized scRNA-seq expresson data  to generate time-course normalized emperical PDF for each TF-candidate gene pair, and save it in `<data generation folder name>` folder.

>>>#2, `gene name reference list` is the list that converts sc expression data gene set into gene symbol IDs. Format: `'gene symbol IDs\t sc gene ID'`.

>>>#3, `TF-target pair list` is the list that contains gene pairs and their labels. format : `'GeneA    GeneB    label'`.

>>>#4, `TF-target pair count list` is a number list that divides gene_pair_list into small parts, where each part corresponds to one TF.

#By file4, we are able to evaluate TDL's performance on only one TF.

>>>#6, `time-course scRNA-seq path+file prefix`  it should be a list of h5 format files, with name of 'prefix0.h5','prefix1.h5'.......

>>>#7， `flag`, 0 means do not generate label list; 1 means to generate label list.

>>#8, `data generation folder name` is a new folder will be created to save generated train and test data.


#################OUTPUT

It generates a 3D NEPDF_data folder, and a series of data files containing `Nxdata_tf` (3D NEPDF file) ,`ydata_TF`(label file) and `zdata_tf` (gene symbol pair file) for each data part divided.

Here we use gene symbol information to align scRNA-seq and gene pair's gene sets. The scRNA-seq may use entrez ID or ensembl ID, or other IDs, gene pair list for GTRD used gene symbol ID, thus we generated `gene name list` to convert all the IDs to gene symbols. Please also do IDs convertion if users want to use their own expression data.

## The file path of 'data_path'in the following code is used in the author's own machine, so please change it to the approprite path according to user's own environment.
>>## 3.2 Training and test 3D CNN model 

    python 3D_CNN.py
    
  `3D_CNN.py` uses `3D NEPDF` and `ground truth` list to train and test `3D CNN` model in three fold cross validation. Please change 'data_path' accordingly.
  
 >>## 3.3 Training and test LSTM model
 
     python conv_lstm_deep.py
     
   `conv_lstm_deep.py` uses `3D NEPDF` and `ground truth list` to train and test `deep conv lstm` models in three fold cross validation. Please change 'data_path' accordingly.
   
 >>## 3.4 Get performance of optimal model for three fold cross validation
  
     python cross_validation_summary.py
     
   `cross_validation_summary.py` collects and summrzies results of the model in each fold to present the final results. Please change 'data_path', 'save_dir', 'the path of trained model' accordingly.
 
 >>## 3.5 Get prediction using the trained model.
  
     python conv_lstm_deep_prediction.py
 >>>### Usage: 

    python conv_lstm_deep_prediction.py <model_path> <newNEPDF_path> 
    
>>>### command line in author's linux machine :

    python conv_lstm_deep_prediction.py /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/new_Xlr00001_KEGG_3d_conv_ddeeper_NT_p600_e100 /home/yey3/nn_project2/data/hesc_2_GSE75748_firstone/TF_target_prediction/GTRD_NT_8X8_6 
   `conv_lstm_deep_prediction.py` uses the trained model to give prediction for new gene pair list. Please change 'data_path', 'save_dir', 'the path of trained model' accordingly.
