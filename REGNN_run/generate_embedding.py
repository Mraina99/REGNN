from skimage import io
import cv2
import numpy as np
from sklearn.decomposition import PCA
import argparse
import scanpy as sc
import random, torch
import pandas as pd
import os
import shutil
import anndata



def generate_embedding_sc(anndata, sample, scgnnsp_dist, scgnnsp_alpha, scgnnsp_k, scgnnsp_zdim, scgnnsp_bypassAE, geneSelectnum, REGNN_method):
    GNN_folder = "REGNN_run/GNN_space/"
    if not os.path.exists(GNN_folder):
        os.makedirs(GNN_folder)
    datasetName = sample+'_'+scgnnsp_zdim+'_'+scgnnsp_alpha+'_'+scgnnsp_k+'_'+scgnnsp_dist+'_logcpm'
    REGNN_data_folder = GNN_folder + datasetName + '/'
    if not os.path.exists(REGNN_data_folder):
        os.makedirs(REGNN_data_folder)
    coords_list = [list(t) for t in zip(anndata.obs["array_row"].tolist(), anndata.obs["array_col"].tolist())]
    if not os.path.exists(REGNN_data_folder + 'coords_array.npy'):
        np.save(REGNN_data_folder + 'coords_array.npy', np.array(coords_list))
    #original_cpm_exp = anndata.X.A.T <----- USE THIS FOR BRAIN DATA
    original_cpm_exp = anndata.X.T #<----- USE THIS FOR KIDNEY DATA
    # original_cpm_exp = pd.read_csv('scGNNsp_space/151507_logcpm_test/151507_human_brain_ex.csv', index_col=0).values
    if not os.path.exists(REGNN_data_folder + sample + '_logcpm_expression.csv'):
        pd.DataFrame(original_cpm_exp).to_csv(REGNN_data_folder + sample + '_logcpm_expression.csv')
    os.chdir(GNN_folder)
    command_preprocessing = 'python -W ignore PreprocessingscGNN.py --datasetName ' + sample + '_logcpm_expression.csv --datasetDir ' + datasetName + '/ --LTMGDir ' + datasetName + '/ --filetype CSV --cellRatio 1.00 --geneSelectnum ' + geneSelectnum + ' --transform None'
    if not os.path.exists(datasetName + '/Use_expression.csv'):
        os.system(command_preprocessing)
    # python -W ignore PreprocessingscGNN.py --datasetName 151507_human_brain_ex.csv --datasetDir 151507_velocity/ --LTMGDir 151507_velocity/ --filetype CSV --cellRatio 1.00 --geneSelectnum 2000 --transform None
    REGNN_output_folder = 'outputdir-3S-' + datasetName + '_EM1_resolution0.3_' + scgnnsp_dist + '_dummy_add_PEalpha' + scgnnsp_alpha + '_k' + scgnnsp_k +'_zdim' + scgnnsp_zdim+ '_NA/'  
    REGNN_output_embedding_csv = datasetName + '_' + scgnnsp_k + '_' + scgnnsp_dist + '_NA_dummy_add_' + scgnnsp_alpha + '_intersect_160_GridEx19_embedding.csv'                                                                                                                                             #GaeHidden 3....                                             dummy
    command_REGNN = 'python -W ignore REGNN_scGNN.py --datasetName ' + datasetName + ' --datasetDir ./  --outputDir ' + REGNN_output_folder + ' --resolution 0.3 --nonsparseMode --EM-iteration 1 --useSpatial --model PAE --useGAEembedding --saveinternal --no-cuda --debugMode savePrune --saveinternal --GAEhidden2 16 --prunetype spatialGrid --PEtypeOp add --pe-type dummy --embedding ' + REGNN_method
    command_REGNN = command_REGNN + " --knn-distance " + scgnnsp_dist
    command_REGNN = command_REGNN + " --PEalpha " + scgnnsp_alpha
    command_REGNN = command_REGNN + " --k " + scgnnsp_k
    command_REGNN = command_REGNN + " --zdim " + scgnnsp_zdim
 
    if not os.path.exists(REGNN_output_folder + REGNN_output_embedding_csv):
        os.system(command_REGNN)
    REGNN_output_embedding = pd.read_csv(REGNN_output_folder + REGNN_output_embedding_csv, index_col=0).values
    if os.path.exists(sample + '_' + scgnnsp_zdim + '_' + scgnnsp_alpha + '_' + scgnnsp_k + '_' + scgnnsp_dist + '_logcpm'):
        shutil.rmtree(sample + '_' + scgnnsp_zdim + '_' + scgnnsp_alpha + '_' + scgnnsp_k + '_' + scgnnsp_dist + '_logcpm')
    if os.path.exists(REGNN_output_folder):
        shutil.rmtree(REGNN_output_folder)
    os.chdir(os.path.dirname(os.getcwd()))
    return REGNN_output_embedding

