import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
from skimage import io, color
from REGNN_run.generate_embedding import generate_embedding_sc
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
    directory_path = args.load_dataset_dir

    #Select Method
    method = args.select_method
    
    # Load in counts and coords
    data_path= os.path.join(directory_path, sample, args.load_count_matrix)
    spatial_path= os.path.join(directory_path, sample, args.load_spatial)
    
    # Number of input genes after preprocessings
    gene_num = str(args.preprocess_top_gene_select)
    
    adata=load_data(data_path,spatial_path)[0]
    
    # --------------------logcpm
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)

    if args.load_annotation_bool == True:
        print("Loading Annoatation file.")
        label_path= os.path.join(directory_path, sample, args.annotation_file)
        label_indexs,n_cluster=load_label(label_path)
        print("n_cluster",n_cluster)
        adata.obs['label'] = label_indexs
    else:
        n_cluster = args.n_clusters
        print("n_cluster",n_cluster)


    print('load data finish')

    sample_nick = sample + "_SSL"
    scgnnsp_bypassAE = scgnnsp_bypassAE_List[1]  # false
    scgnnsp_usePCA = scgnnsp_usePCA_List[1]  # false
    optical_img_path = None
    ARI_list, AMI_list, CHS_list, DBS_list, CS_list, CM_list, PCM_list, FMS_list, HCVM_list, HS_list, MI_list, NMI_list, RI_list, SSc_list, SSa_list, VMS_list  = ([]  for _ in range(16))
    name_list = []
    for scgnnsp_dist in scgnnsp_knn_distanceList:
            for scgnnsp_k in scgnnsp_kList:
                    all_results_path = f"result/all_results_{method}.csv"
                    ARI_table_path = f"result/ARI_table_{method}.csv"
                    # --------------------------------------------------------------------------------------------------------#
                    # -------------------------------generate_embedding --------------------------------------------------#
                    image_name = sample_nick + '_scGNN_' + 'logcpm' + '_PEalpha' + str(scgnnsp_alpha) + '_zdim' + str(
                        scgnnsp_zdim)
                    column_name = 'PEalpha' + str(scgnnsp_alpha) + '_zdim' + str(
                        scgnnsp_zdim)
                    embedding = generate_embedding_sc(adata, sample=sample, scgnnsp_dist=scgnnsp_dist, scgnnsp_alpha=scgnnsp_alpha, scgnnsp_k=scgnnsp_k,scgnnsp_zdim=scgnnsp_zdim, scgnnsp_bypassAE=scgnnsp_bypassAE, geneSelectnum=gene_num, REGNN_method = method)
                    adata.obsm["embedding"] = embedding
                    
                    # Export final embeddings
                    np.savetxt('result/Embeddings/'+image_name+'_embedding.csv', adata.obsm["embedding"], delimiter=',')
                    
                    print('Embeddings output to REGNN_run/result/Embeddings folder')

                    # Get clustering from kmeans using embeddings
                    kmeans(adata, n_cluster)
                    layer = adata.obs["layer"]
                    np.savetxt('result/Cluster_labels/'+image_name+'_clusters.csv', adata.obs["layer"], delimiter=',')

                    print('Clusterings output to REGNN_run/result/Cluster_labels folder')


                    if args.output_ARI == True:
                        if args.load_annotation_bool == True:
 
                            ARI, AMI, CHS, DBS, CS, CM, PCM, FMS, HCVM, HS, MI, NMI, RI, SSc, SSa, VMS = calculate(adata)
                            
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
                                if not os.path.exists(all_results_path):
                                    ARI_result.to_csv(all_results_path, index=True, header=True,mode = 'a')
                                else: 
                                    ARI_result.to_csv(all_results_path, index=True, header=False,mode = 'a')
                                
                                table = pd.read_csv(ARI_table_path, index_col=0)
                                table.at[sample,column_name] = ARI
                                table.to_csv(ARI_table_path, index=True, header=True, mode = 'w')
                                
                                return
                            
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
                            
                            # Used to print ARI results if annotation file is included & output_ARI = True
                            if not os.path.exists('result/'):
                                os.makedirs('result/')
                            if not os.path.exists(all_results_path):
                                ARI_result.to_csv(all_results_path, index=True, header=True,mode = 'a')
                            else: 
                                ARI_result.to_csv(all_results_path, index=True, header=False,mode = 'a')
                            
                            table = pd.read_csv(ARI_table_path, index_col=0)
                            table.at[sample,column_name] = ARI
                            table.to_csv(ARI_table_path, index=True, header=True, mode = 'w')
                        else:
                            print("Annotation/Ground truth labels not included. Cannot compute and export ARI.")
                    
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




def initialize_REGNN(args):
    # args = parse_args()
    scgnnsp_zdimList = [str(x) for x in args.zdim]  
    scgnnsp_PEalphaList = [str(x) for x in args.PEalpha]  
    ARI_full_table_path = f"REGNN_run/result/ARI_table_{args.select_method}.csv"

    # Testing sample(s)
    samples = ["V10S14-085_XY04_21-0057"]
    samples = [args.load_dataset_name]

    if args.output_ARI == True:
        df = pd.DataFrame(index=pd.Index(samples))
        for i in scgnnsp_zdimList:
            for j in scgnnsp_PEalphaList:
                column_name = 'PEalpha' + str(j) + '_zdim' + str(i)
                df[column_name] = None
        df.to_csv(ARI_full_table_path, index=True, header=True, mode = 'w')


    # core_num = cpu_count()
    pool = Pool(64)
    # Loop through different PE & zdims and get REGNN embedding and clusterings
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
