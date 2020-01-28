# Time-course Deep Learning (TDL) models for gene interaction inference
# Title, Deep learning of gene interactions from single cell time-course expression data
# https://www.biorxiv.org/content/
## date: 28 Jan, 2020

># 1, TDL overview
![](https://github.com/xiaoyeye/TDL/blob/master/TDL_overview.bmp)

Time-course deep learning (TDL) model architecture. To infer gene interactions (top left) we first convert time-course single cell expression data to a 3D tensor, which we term normalized empirical probability distribution function (NEPDF). Each 2D slice of the NEPDF captures the co-expression of a pair of genes at one of the time points profiled and the 3D NEPDF represents their co-expression over time. 3D NEPDF is then used as input to atime-course deep learning (TDL) model. The model is trained using labeled positive and negative pairs. The figure shows the convolutional Long short-term memory (LSTM) architecture which is one of the two TDL model we tested. This mode consists of two stacked LSTM layers, followed by a dense layer which concatenates all convolutional hidden state from LSTM layer and then a final output (classifica-tion) layer. See Supporting Figure S1 for the other TDL architecture we tested, 3D CNN.

># 2, Code environment

>>## Users need to install python and ‘Keras’ and ‘theano’ modules, and  all ohther modules required by the code. We  recommend Anaconda to do this.
Author's environment is python 3.6.3 in a Linux server which is now running Centos 6.5 as the underlying OS and Rocks 6.1.1 as the cluster management revision. 
##    ############################## still working on the following ...
># 3, Example for running in interaction task for 
Users should first set the path as the downloaded folder. 
>>## 3.1 Training and test data generation 
>>>### Usage: 

    python get_xy_label_data_cnnc_combine_from_database_3d8X8.py <None> <gene name list> <TF-target pair list> <TF-target pair count list> <None> <time-course scRNA-seq path+file prefix> <time point number> <flag if have ground truth in pair list> <data generation folder name> 


`data_generation_interaction_ten_fold.py` uses the spatial location data to generate normalized adjacent matrix of cells, and save it in `seqfish_plus` folder; also uses the expression data to generate expression matrix for ten fold cross validation, and save it in `rand_1_10fold` folder.

>>## 3.2 Training and test model

    python gcn_LR2_LR_as_nega_big.py
    
  `gcn_LR2_LR_as_nega_big.py` uses normalized adjacent matrix to generate normalized laplacian matrix, and then uses laplacian matrix to train and test GCNG models in ten fold cross validation. 
  
 >>## 3.3 get optimal model
 
     python gcn_LR2_LR_as_nega_big_layer_predict_min.py
     
   `gcn_LR2_LR_as_nega_big_layer_predict_min.py` tries to find the optimal model during the trainning, by monitoring the validation dataset's accuracy.
   
 >>## 3.4 get performance of optimal model for ten fold cross validation
  
     python predict_analysis_more_kegg_tfs_average_whole_new_rand.py
     
   `predict_analysis_more_kegg_tfs_average_whole_new_rand.py` collects results of the optimal model in each fold to present the final results.
 
