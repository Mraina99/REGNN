import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
from skimage import io, color
#from REGNN_GAE.pipeline_transform_embedding_to_image import get_ground_truth, get_clusters
from REGNN_GAE.generate_embedding import generate_embedding_sc
import random, torch
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
    
 
    for i in label_list['class']: 
        if i not in label_names:
            label_indexs.append(len(label_names))
            label_names.append(i)
        else:
            label_indexs.append(label_names.index(i))
    #print("Label Names: ", label_names)
    #print("Label Indexes: ", label_indexs)
    return label_indexs,len(label_names)

def pseudo_images(scgnnsp_zdim,scgnnsp_alpha,sample, args):
    print(sample)
    scgnnsp_knn_distanceList = ['euclidean']
    scgnnsp_kList = ['6']
    scgnnsp_bypassAE_List = [True, False]
    scgnnsp_usePCA_List = [True, False]

    
    #Change data directory as needed
    directory_path = "/N/slate/mraina/Juexin/Eadon/newkidneydata/"
    directory_path = args.load_dataset_dir
    
    data_path= os.path.join(directory_path, sample, args.load_count_matrix)
    spatial_path= os.path.join(directory_path, sample, args.load_spatial)
    label_path= os.path.join(directory_path, sample, args.load_annotation)
    #output_folder = "kidney_results_1/RGB_images/M32/"
    
    
    adata=load_data(data_path,spatial_path)[0]
    
    # print(adata.shape)

    # --------------------logcpm
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)

    label_indexs,n_cluster=load_label(label_path)
    print("n_cluster",n_cluster)
    # print(len(label_indexs))
    adata.obs['label'] = label_indexs

    print('load data finish')

    sample_nick = sample + "_GAE"
    scgnnsp_bypassAE = scgnnsp_bypassAE_List[1]  # false
    scgnnsp_usePCA = scgnnsp_usePCA_List[1]  # false
    optical_img_path = None
    ARI_list, AMI_list, CHS_list, DBS_list, CS_list, CM_list, PCM_list, FMS_list, HCVM_list, HS_list, MI_list, NMI_list, RI_list, SSc_list, SSa_list, VMS_list  = ([]  for _ in range(16))
    name_list = []
    for scgnnsp_dist in scgnnsp_knn_distanceList:
            for scgnnsp_k in scgnnsp_kList:
                    # --------------------------------------------------------------------------------------------------------#
                    # -------------------------------generate_embedding --------------------------------------------------#
                    image_name = sample_nick + '_scGNN_' + 'logcpm' + '_PEalpha' + str(scgnnsp_alpha) + '_zdim' + str(
                        scgnnsp_zdim)
                    column_name = 'PEalpha' + str(scgnnsp_alpha) + '_zdim' + str(
                        scgnnsp_zdim)
                    embedding = generate_embedding_sc(adata, sample=sample, scgnnsp_dist=scgnnsp_dist, scgnnsp_alpha=scgnnsp_alpha, scgnnsp_k=scgnnsp_k,scgnnsp_zdim=scgnnsp_zdim, scgnnsp_bypassAE=scgnnsp_bypassAE)
                    # embedding = embedding.detach().numpy()
                    #embedding = pd.read_csv('result/Embeddings/'+image_name+'_embedding_GAEfinal.csv',names =['embedding0', 'embedding1', 'embedding2'], header=None)
                    #print(embedding.shape) 
                    #print(embedding) 
                    # embedding = embedding_data.loc[:, ['embedding0', 'embedding1', 'embedding2']].values
                    adata.obsm["embedding"] = embedding
                    #np.savetxt('embedding/'+image_name+'_embedding.csv', embedding, delimiter=',')
                    np.savetxt('result/Embeddings/'+image_name+'_embedding.csv', adata.obsm["embedding"], delimiter=',')
                    
                    print('generate embedding finish')

                    if (embedding != embedding).any() == True:
                        print("The embedding contains NaNs. Exiting iteration.")
                        ARI = "NaN"
                        ARI_list.append(ARI)
                        name_list.append(image_name)
                        ARI_result = {
                            'name': name_list,
                            'ARI': ARI_list,
                        }
                        ARI_result = pd.DataFrame(ARI_result)
                        
                        if not os.path.exists('result/'):
                            os.makedirs('result/')
                        if not os.path.exists('result/all_results_GAE.csv'):
                            ARI_result.to_csv('result/all_results_GAE.csv', index=True, header=True,mode = 'a')
                        else: 
                            ARI_result.to_csv('result/all_results_GAE.csv', index=True, header=False,mode = 'a')
                        
                        table = pd.read_csv('result/ARI_table_GAE.csv', index_col=0)
                        table.at[sample,column_name] = ARI
                        table.to_csv('result/ARI_table_GAE.csv', index=True, header=True, mode = 'w')
                        
                        return

                    # --------------------------------------------------------------------------------------------------------#
                    # --------------------------------transform_embedding_to_image-------------------------------------------------#
                    """
                    transform_embedding_to_image(adata, image_name, output_folder,
                                                 img_type='lowres',
                                                 scale_factor_file=True)  # img_type:lowres,hires,both
                    print('transform embedding to image finish')

                    get_ground_truth(adata, sample, output_folder,
                                                 img_type='lowres',
                                                 scale_factor_file=True)  # img_type:lowres,hires,both
                    print('ground truth to image finish')
                    """
                    """
                    print("n_cluster", n_cluster) # originally print("n_cluster",3)
                    for n_cluster in range(1,n_cluster):
                        kmeans(adata, n_cluster)
                        print(n_cluster)
                        ARI = calculate(adata)
                    """
                    # if (sample != '151669' or sample != '151670' or sample != '151671' or sample != '151672' or sample != '2-8'):
                    #     kmeans(adata, 7)
                    # elif (sample == '2-8'):
                    #     kmeans(adata, 6)
                    # else:
                    #     print("n_cluster",n_cluster)
                    #     kmeans(adata, n_cluster)
                    kmeans(adata, n_cluster)
                    ARI, AMI, CHS, DBS, CS, CM, PCM, FMS, HCVM, HS, MI, NMI, RI, SSc, SSa, VMS = calculate(adata)

                    layer = adata.obs["layer"]
                    np.savetxt('result/Cluster_labels/'+image_name+'_clusters.csv', adata.obs["layer"], delimiter=',')

                    # Print clusters
                    """
                    get_clusters(adata, image_name, output_folder,
                                                 img_type='lowres',
                                                 scale_factor_file=True)  # img_type:lowres,hires,both
                    print('transform clusters to image finish')
                    """
                    
                    ARI_list.append(ARI)
                    AMI_list.append(AMI)
                    CHS_list.append(CHS)
                    DBS_list.append(DBS)
                    CS_list.append(CS)
                    CM_list.append(CM)
                    PCM_list.append(PCM)
                    FMS_list.append(FMS)
                    HCVM_list.append(HCVM)
                    HS_list.append(HS)
                    MI_list.append(MI)
                    NMI_list.append(NMI)
                    RI_list.append(RI)
                    SSc_list.append(SSc)
                    SSa_list.append(SSa)
                    VMS_list.append(VMS)
                    name_list.append(image_name)
                    ARI_result = {
                        'name': name_list,
                        'ARI': ARI_list,
                        'AMI': AMI_list, 
                        'CHS': CHS_list, 
                        'DBS': DBS_list, 
                        'CS': CS_list, 
                        'CM': CM_list, 
                        'PCM': PCM_list, 
                        'FMS': FMS_list, 
                        'HCVM': HCVM_list, 
                        'HS': HS_list, 
                        'MI': MI_list, 
                        'NMI': NMI_list, 
                        'RI': RI_list, 
                        'SSc': SSc_list, 
                        #'SSa': SSa_list, 
                        'VMS': VMS_list 
                    }
                    ARI_result = pd.DataFrame(ARI_result)
                    
                    # adata.obs.to_csv('category_map/'+sample+'/'+image_name+'_meta.csv')
                    if not os.path.exists('result/'):
                        os.makedirs('result/')
                    #ARI_result.to_csv('result/'+sample + '_result.csv', index=True, header=True,mode = 'a')
                    if not os.path.exists('result/all_results_GAE.csv'):
                        ARI_result.to_csv('result/all_results_GAE.csv', index=True, header=True,mode = 'a')
                    else: 
                        ARI_result.to_csv('result/all_results_GAE.csv', index=True, header=False,mode = 'a')
                    
                    table = pd.read_csv('result/ARI_table_GAE.csv', index_col=0)
                    table.at[sample,column_name] = ARI
                    table.to_csv('result/ARI_table_GAE.csv', index=True, header=True, mode = 'w')
                    
                    os.getcwd()

def kmeans(adata, n):

    clf = KMeans(n_clusters=n)
    y_pred = clf.fit_predict(adata.obsm["embedding"])
    adata.obs['layer'] = y_pred
    return adata

def calculate(adata):
    label = adata.obs['label'].values
    print("Label: ", adata.obs['label'].values, "Length: ", len(label))
    layer = adata.obs['layer'].values
    print("Layer: ", adata.obs['layer'].values, "Length: ", len(layer))
    
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




def calc_ARI_GAE(args):
    # args = parse_args()
    scgnnsp_zdimList = ['3','8', '64', '128' ,'256', '512']
    scgnnsp_PEalphaList =  ['1.0', '2.0', '3.0', '4.0', '4.5']


    # Test all samples
    """
    samples = []
    """

    # Testing one sample (Current code is set up for one sample)
    samples = ["V10S14-085_XY04_21-0057"]
    samples = [args.load_dataset_name]

    df = pd.DataFrame(index=pd.Index(samples))
    for i in scgnnsp_zdimList:
        for j in scgnnsp_PEalphaList:
            column_name = 'PEalpha' + str(j) + '_zdim' + str(i)
            df[column_name] = None
    df.to_csv('REGNN_GAE/result/ARI_table_GAE.csv', index=True, header=True, mode = 'w')


    # core_num = cpu_count()
    pool = Pool(64)
    # for sample in sample_list:
        # pseudo_image_folder = output_path + "/" + sample + "/"
        # if not os.path.exists(pseudo_image_folder):
        #     os.makedirs(pseudo_image_folder)
    print("Start Main loop")
    for samp in samples:
        for scgnnsp_zdim in scgnnsp_zdimList:
            for scgnnsp_alpha in scgnnsp_PEalphaList:
                tmp=pool.apply_async(pseudo_images,(scgnnsp_zdim,scgnnsp_alpha,samp, args))
                    # pseudo_images(scgnnsp_zdim, scgnnsp_alpha, sample)
                tmp.get()
                #break
            #break
    pool.close()
    pool.join()

