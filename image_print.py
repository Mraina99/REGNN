import pandas as pd
#from pipeline_sparse_expression_to_image import save_transformed_RGB_to_image_and_csv, scale_to_RGB
import json, os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os,csv,re,json
import scanpy as sc
import math
# import SpaGCN as spg
#import cv2
from skimage import io, color
# from SpaGCN.util import prefilter_specialgenes, prefilter_genes
from pipeline_transform_spaGCN_embedding_to_image_backup import transform_embedding_to_image, get_ground_truth, get_clusters
from generate_embedding import generate_embedding_sc
# from util import prefilter_genes, prefilter_specialgenes
import random, torch
# from test import segmentation
# from inpaint_images import inpaint
import warnings
from multiprocessing import Pool, cpu_count
import argparse
from functools import partial
import shutil
import os
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as skl
# warnings.filterwarnings("ignore")


def load_data(h5_path, spatial_path, scale_factor_path=None):
    # new
    # Read in gene expression and spatial location
    # adata = sc.read_10x_h5(h5_path)
    adata=sc.read_csv(h5_path)
    spatial_all = pd.read_csv(spatial_path, sep=",", 
                              header=0, 
                              na_filter=False)

    adata.obs["in_tissue"] = 1
    adata.obs["array_row"] = spatial_all['x'].tolist()
    adata.obs["array_col"] = spatial_all['y'].tolist()
    adata.obs["pxl_col_in_fullres"] = 1
    adata.obs["pxl_row_in_fullres"] = 1
    adata.obs.index.name = 'barcode'
    adata.var_names_make_unique()

    # Read scale_factor_file
    adata.uns["spot_diameter_fullres"] = 1
    adata.uns["tissue_hires_scalef"] = 1
    adata.uns["fiducial_diameter_fullres"] = 1
    adata.uns["tissue_lowres_scalef"] = 1

    return adata , spatial_all


def load_label(label_path):
    # --------------------label
    
    label_list = pd.read_csv(label_path)
    label_names=[]
    label_indexs=[]
    label_names_str=[]
    
    #print(label_list)
    # print()
    # print(label_list['transfer_subset'])
    # transfer_subset,l1_label,class_label,subclass_label
    for i in label_list['class']: #for i in label_list['class_label']:
        if i not in label_names:
            label_indexs.append(len(label_names))
            label_names.append(i)
            label_names_str.append(i)
        else:
            label_indexs.append(label_names.index(i))
            label_names_str.append(i)
    #print("Label Names: ", label_names)
    #print("Label Indexes: ", label_indexs)
    # print(adata.obs)
    return label_list,label_indexs,len(label_names),label_names_str

def pseudo_images(scgnnsp_zdim,scgnnsp_alpha,sample):
    print(sample)
    scgnnsp_knn_distanceList = ['euclidean']
    scgnnsp_kList = ['6']
    scgnnsp_bypassAE_List = [True, False]
    scgnnsp_usePCA_List = [True, False]
    

    directory_path = "/N/slate/mraina/Juexin/Eadon/kidneydata/"
    data_path= os.path.join(directory_path, sample, "count.csv")
    spatial_path= os.path.join(directory_path, sample, "spa.csv")
    label_path= os.path.join(directory_path, sample, "labels.csv")
    output_folder = "/N/slate/mraina/REGNN/kidney_results_1/RGB_images/"
    
    
    adata=load_data(data_path,spatial_path)[0]
    
    # --------------------logcpm
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)

    label_list,label_indexs,n_cluster,label_names=load_label(label_path)
    print("n_cluster",n_cluster)
    # print(len(label_indexs))
    adata.obs['label'] = label_names
    #print(adata.obs['label'])

    sample_nick = sample.partition('_')[0] + "_" + sample.partition('-')[2]
    image_name = sample_nick + '_scGNN_' + 'logcpm' + '_PEalpha' + str(scgnnsp_alpha) + '_zdim' + str(scgnnsp_zdim)
    clusters = pd.read_csv('/N/slate/mraina/REGNN/result/Cluster_labels/kidney085_0057_scGNN_logcpm_PEalpha2.0_zdim256_clusters.csv', header=None)
    #clusters = sc.read('/N/slate/mraina/REGNN/result/Cluster_labels/results.h5ad')
    clusters["layers"] = clusters.iloc[:, [0]].astype(str)
    #print(clusters["layers"])
    
    clusters["layers"].replace("0.0", "epithelial cells", inplace=True)
    clusters["layers"].replace("1.0", "endothelial cells", inplace=True)
    clusters["layers"].replace("2.0", "stroma cells", inplace=True)
    clusters["layers"].replace("3.0", "immune cells", inplace=True)
    clusters["layers"].replace("4.0", "neural cells", inplace=True)

    #print(clusters["layers"])
    adata.obs['layer'] = clusters["layers"].values
    print(adata.obs['layer'])

    print('load data finish')

    ARI, AMI, CHS, DBS, CS, CM, PCM, FMS, HCVM, HS, MI, NMI, RI, SSc, SSa, VMS = calculate(adata)

    print("ARI: ", ARI)
    print("NMI: ", NMI)
    print("RI: ", RI)
    print("FMS: ", FMS)
    print("DBS: ", DBS)


    get_ground_truth(adata, sample, output_folder,
        img_type='lowres',
        scale_factor_file=True)  # img_type:lowres,hires,both
    print('ground truth to image finish')

    get_clusters(adata, image_name, output_folder,
        img_type='lowres',
        scale_factor_file=True)  # img_type:lowres,hires,both
    print('transform clusters to image finish')

def calculate(adata):
    label = adata.obs['label'].values
    #print("Label: ", adata.obs['label'].values, "Length: ", len(label))
    layer = adata.obs['layer'].values
    #print("Layer: ", adata.obs['layer'].values, "Length: ", len(layer))

    ARI = skl.adjusted_rand_score(label, layer)
    AMI = skl.adjusted_mutual_info_score(label, layer)
    CHS = skl.calinski_harabasz_score(adata.X, layer)
    DBS = skl.davies_bouldin_score(adata.X, layer)
    CS = skl.completeness_score(label, layer)
    CM = skl.contingency_matrix(label, layer)
    PCM = skl.pair_confusion_matrix(label, layer)
    FMS = skl.fowlkes_mallows_score(label, layer)
    HCVM = skl.homogeneity_completeness_v_measure(label, layer)
    HS = skl.homogeneity_score(label, layer)
    MI = skl.mutual_info_score(label, layer)
    NMI = skl.normalized_mutual_info_score(label, layer)
    RI = skl.rand_score(label, layer)
    SSc = skl.silhouette_score(adata.X, layer)
    SSa = skl.silhouette_samples(adata.X, layer)
    VMS = skl.v_measure_score(label, layer)
    print(ARI)
    return ARI, AMI, CHS, DBS, CS, CM, PCM, FMS, HCVM, HS, MI, NMI, RI, SSc, SSa, VMS

if __name__ == '__main__':
    scgnnsp_zdimList = ['3','10', '16','32', '64', '128', '256']
    scgnnsp_PEalphaList = ['0.1','0.2','0.3', '0.5', '1.0', '1.2', '1.5','2.0']

    samples = ['kidney085_XY01_20-0038',
                'kidney085_XY02_20-0040',
                'kidney085_XY03_21-0056',
                'kidney085_XY04_21-0057',
                'kidney086_XY01_21-0055',
                'kidney086_XY02_20-0039',
                'kidney086_XY03_21-0063',
                'kidney086_XY04_21-0066',
                'kidney087_XY01_21-0061',
                'kidney087_XY02_21-0063',
                'kidney087_XY03_21-0064',
                'kidney087_XY04_21-0065',
                'kidney102_XY02_IU-21-019-5',
                'kidney102_XY03_IU-21-015-2',
                'kidney388_XY01_21-0068',
                'kidney388_XY02_20-0071',
                'kidney388_XY03_20-0072',
                'kidney388_XY04_20-0073',
                'kidney016_XY01_18-0006',
                'kidney017_XY03-13437',
                'kidney019_XY02-M32',
                'kidney019_XY03-M61',
                'kidney019_XY04-F52']


    pseudo_images('256','2.0',"kidney085_XY04_21-0057")

