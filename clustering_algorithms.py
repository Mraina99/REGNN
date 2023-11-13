import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
# import SpaGCN as spg
#import cv2
from skimage import io, color
# from SpaGCN.util import prefilter_specialgenes, prefilter_genes
from pipeline_transform_spaGCN_embedding_to_image import transform_embedding_to_image, get_ground_truth, get_clusters
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
import SpaGCN as spg
import cv2
import os
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
# warnings.filterwarnings("ignore")


def load_data(h5_path, spatial_path, scale_factor_path=None):
    # new
    # Read in gene expression and spatial location
    # adata = sc.read_10x_h5(h5_path)
    adata=sc.read_csv(h5_path)
    spatial_all = pd.read_csv(spatial_path, sep=",", 
                              header=0, 
                              na_filter=False)
    img_data = cv2.imread("histology.tif")

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

    # histology image data
    x_array=spatial_all[2].tolist()
    y_array=spatial_all[3].tolist()
    x_pixel=spatial_all[4].tolist()
    y_pixel=spatial_all[5].tolist()

    img_new=img.copy()
    for i in range(len(x_pixel)):
        x=x_pixel[i]
        y=y_pixel[i]
        img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

    cv2.imwrite('./sample_results/151673_map.jpg', img_new)


    return adata , spatial_all


def load_label(label_path):
    # --------------------label
    
    label_list = pd.read_csv(label_path)
    label_names=[]
    label_indexs=[]
    
    #print(label_list)
    # print()
    # print(label_list['transfer_subset'])
    # transfer_subset,l1_label,class_label,subclass_label
    for i in label_list['class']: #for i in label_list['class_label']:
        if i not in label_names:
            label_indexs.append(len(label_names))
            label_names.append(i)
        else:
            label_indexs.append(label_names.index(i))
    #print("Label Names: ", label_names)
    #print("Label Indexes: ", label_indexs)
    # print(adata.obs)
    return label_indexs,len(label_names)

def pseudo_images(scgnnsp_zdim,scgnnsp_alpha,sample):
    print(sample)
    scgnnsp_knn_distanceList = ['euclidean']
    scgnnsp_kList = ['6']
    scgnnsp_bypassAE_List = [True, False]
    scgnnsp_usePCA_List = [True, False]
    
    data_path = "/scratch/scdata/pipeline/original/"+sample+"/"
    h5_path = sample+"_filtered_feature_bc_matrix.h5"
    scale_factor_path = "spatial/scalefactors_json.json"
    spatial_path = "spatial/tissue_positions_list.csv"
    label_path = 'label_csv/'+sample+'.csv'
    zdim_row = "zdim_" + str(scgnnsp_zdim) 
    PEalpha_col = "PEalpha_" + str(scgnnsp_alpha)
    # pseudo_image_folder = output_path+"/"+sample+"/"s
    # if not os.path.exists(pseudo_image_folder):
    #     os.makedirs(pseudo_image_folder)
    # pseudo_image_folder = output_path


    # --------------------------------------------------------------------------------------------------------#
    # -------------------------------load data--------------------------------------------------#
    # Read in gene expression and spatial location
    # adata = sc.read_10x_h5(os.path.join(data_path,h5_path))
    # # print(adata.X)
    # spatial_all=pd.read_csv(os.path.join(data_path,spatial_path),sep=",",header=None,na_filter=False,index_col=0)
    # spatial = spatial_all[spatial_all[1] == 1]
    # spatial = spatial.sort_values(by=0)
    # assert all(adata.obs.index == spatial.index)
    # adata.obs["in_tissue"]=spatial[1]
    # adata.obs["array_row"]=spatial[2]
    # adata.obs["array_col"]=spatial[3]
    # adata.obs["pxl_col_in_fullres"]=spatial[4]
    # adata.obs["pxl_row_in_fullres"]=spatial[5]
    # adata.obs.index.name = 'barcode'
    # adata.var_names_make_unique()


    # # Read scale_factor_file
    # with open(os.path.join(data_path,scale_factor_path)) as fp_scaler:
    #     scaler = json.load(fp_scaler)
    # adata.uns["spot_diameter_fullres"] = scaler["spot_diameter_fullres"]
    # adata.uns["tissue_hires_scalef"] = scaler["tissue_hires_scalef"]
    # adata.uns["fiducial_diameter_fullres"] = scaler["fiducial_diameter_fullres"]
    # adata.uns["tissue_lowres_scalef"] = scaler["tissue_lowres_scalef"]
    # 038
    
    directory_path = "/data/"
    
    data_path= os.path.join(directory_path, sample, "count.csv")
    spatial_path= os.path.join(directory_path, sample, "spa.csv")
    label_path= os.path.join(directory_path, sample, "labels.csv")    
    
    adata=load_data(data_path,spatial_path)[0]
    
    # print(adata.shape)

    # --------------------logcpm
    sc.pp.normalize_total(adata,target_sum=1e4)
    sc.pp.log1p(adata)

    label_indexs,n_cluster=load_label(label_path)
    print("n_cluster",n_cluster)
    # print(len(label_indexs))
    adata.obs['label'] = label_indexs

    #print("adata object: ", adata.obs['label'])
    #lol = adata.obs['label']
    #print("out of object: ", lol)
    #print(lol[1])

    print('load data finish')

    sample_nick = sample.partition('_')[0] + "_" + sample.partition('-')[2]
    scgnnsp_bypassAE = scgnnsp_bypassAE_List[1]  # false
    scgnnsp_usePCA = scgnnsp_usePCA_List[1]  # false
    # optical_img_path = os.path.join(data_path,"spatial/tissue_hires_image.png")
    optical_img_path = None
    ARI_list = []
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
                    # embedding_data = pd.read_csv("embedding.csv",names =['embedding0', 'embedding1', 'embedding2'])
                    # embedding = embedding_data.loc[:, ['embedding0', 'embedding1', 'embedding2']].values
                    adata.obsm["embedding"] = embedding
                    #np.savetxt('embedding/'+sample+'/'+image_name+'_embedding.csv', embedding, delimiter=',')

                    #np.savetxt('/N/slate/mraina/RESEPT_EGNN/result/Embeddings/'+image_name+'_embedding.csv', adata.obsm["embedding"], delimiter=',')
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
                        if not os.path.exists('result/clustering_results.csv'):
                            ARI_result.to_csv('result/clustering_results.csv', index=True, header=True,mode = 'a')
                        else: 
                            ARI_result.to_csv('result/clustering_results.csv', index=True, header=False,mode = 'a')
                        
                        #table = pd.read_csv('/N/slate/mraina/RESEPT_EGNN/result/bugtest.csv', index_col=0)
                        #table.at[sample,column_name] = ARI
                        #table.to_csv('/N/slate/mraina/RESEPT_EGNN/result/bugtest.csv', index=True, header=True, mode = 'w')

                        
                        table = pd.read_csv('/result/kmeans.csv', index_col=0)
                        table.at[zdim_row, PEalpha_col] = ARI
                        table.to_csv('/result/kmeans.csv', index=True, header=True, mode = 'w')

                        table = pd.read_csv('/result/spectral.csv', index_col=0)
                        table.at[zdim_row, PEalpha_col] = ARI
                        table.to_csv('/result/spectral.csv', index=True, header=True, mode = 'w')

                        table = pd.read_csv('/result/affinity.csv', index_col=0)
                        table.at[zdim_row, PEalpha_col] = ARI
                        table.to_csv('/result/affinity.csv', index=True, header=True, mode = 'w')

                        table = pd.read_csv('/result/agglomerative.csv', index_col=0)
                        table.at[zdim_row, PEalpha_col] = ARI
                        table.to_csv('/result/agglomerative.csv', index=True, header=True, mode = 'w')
                        

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

                    print("Label: ", adata.obs['label'].values, "Length: ", len(adata.obs['label'].values))

                    kmeans(adata, n_cluster)
                    spectral(adata, n_cluster)
                    affinity(adata, n_cluster)
                    agglomerative(adata, n_cluster)

                    #ARI = calculate(adata)
                    print("KMeans: ")
                    ARI_k = calculate(adata.obs['label'].values, adata.obs['layer_kmeans'].values)
                    #np.savetxt('/N/slate/mraina/RESEPT_EGNN/result/Cluster_labels/'+image_name+'_Kmeans.csv', adata.obs["layer_kmeans"], delimiter=',')
                    print("Spectral: ")
                    ARI_s = calculate(adata.obs['label'].values, adata.obs['layer_Spectral'].values)
                    #np.savetxt('/N/slate/mraina/RESEPT_EGNN/result/Cluster_labels/'+image_name+'_Spectral.csv', adata.obs["layer_Spectral"], delimiter=',')
                    print("Affinity: ")
                    ARI_af = calculate(adata.obs['label'].values, adata.obs['layer_Affinity'].values)
                    #np.savetxt('/N/slate/mraina/RESEPT_EGNN/result/Cluster_labels/'+image_name+'_Affinity.csv', adata.obs["layer_Affinity"], delimiter=',')
                    print("Agglomerative: ")
                    ARI_ag = calculate(adata.obs['label'].values, adata.obs['layer_Agglomerative'].values)
                    #np.savetxt('/N/slate/mraina/RESEPT_EGNN/result/Cluster_labels/'+image_name+'_Agglomerative.csv', adata.obs["layer_Agglomerative"], delimiter=',')


                    #layer = adata.obs["layer"]
                    #np.savetxt('/N/slate/mraina/RESEPT_EGNN/result/Cluster_labels/'+image_name+'_clusters.csv', adata.obs["layer"], delimiter=',')

                    # Print clusters
                    """
                    get_clusters(adata, image_name, output_folder,
                                                 img_type='lowres',
                                                 scale_factor_file=True)  # img_type:lowres,hires,both
                    print('transform clusters to image finish')
                    """
                
                    #ARI_list.append(ARI)
                    name_list.append(image_name)
                    ARI_result = {
                        'name': name_list,
                        'kMeans': ARI_k,
                        'Spectral': ARI_s,
                        'Affinity': ARI_af,
                        'Agglomerative': ARI_ag
                    }

                    #ARI_list.append(ARI)
                    #name_list.append(image_name)
                    #ARI_result = {
                    #    'name': name_list,
                    #    'ARI': ARI_list,
                    #}
                    ARI_result = pd.DataFrame(ARI_result)
                    # adata.obs.to_csv('category_map/'+sample+'/'+image_name+'_meta.csv')
                    if not os.path.exists('result/'):
                        os.makedirs('result/')
                    #ARI_result.to_csv('result/'+sample + '_result.csv', index=True, header=True,mode = 'a')
                    if not os.path.exists('result/clustering_results.csv'):
                        ARI_result.to_csv('result/clustering_results.csv', index=True, header=True,mode = 'w')
                    else: 
                        ARI_result.to_csv('result/clustering_results.csv', index=True, header=False,mode = 'a')
                    
                    #table = pd.read_csv('/N/slate/mraina/RESEPT_EGNN/result/bugtest.csv', index_col=0)
                    #table.at[sample,column_name] = ARI
                    #table.to_csv('/N/slate/mraina/RESEPT_EGNN/result/bugtest.csv', index=True, header=True, mode = 'w')

                    
                    table = pd.read_csv('/result/kmeans.csv', index_col=0)
                    table.at[zdim_row, PEalpha_col] = ARI_k
                    table.to_csv('/result/kmeans.csv', index=True, header=True, mode = 'w')

                    table = pd.read_csv('/result/spectral.csv', index_col=0)
                    table.at[zdim_row, PEalpha_col] = ARI_s
                    table.to_csv('/result/spectral.csv', index=True, header=True, mode = 'w')

                    table = pd.read_csv('/result/affinity.csv', index_col=0)
                    table.at[zdim_row, PEalpha_col] = ARI_af
                    table.to_csv('/result/affinity.csv', index=True, header=True, mode = 'w')

                    table = pd.read_csv('/result/agglomerative.csv', index_col=0)
                    table.at[zdim_row, PEalpha_col] = ARI_ag
                    table.to_csv('/result/agglomerative.csv', index=True, header=True, mode = 'w')
                    

                    os.getcwd()

def kmeans(adata, n):

    clf = KMeans(n_clusters=n)
    y_pred = clf.fit_predict(adata.obsm["embedding"])
    adata.obs['layer_kmeans'] = y_pred
    print("Layer: ", adata.obs['layer_kmeans'].values, "Length: ", len(adata.obs['layer_kmeans'].values))
    return adata

def spectral(adata, n):

    clf = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0)
    y_pred = clf.fit_predict(adata.obsm["embedding"])
    adata.obs['layer_Spectral'] = y_pred
    print("Layer: ", adata.obs['layer_Spectral'].values, "Length: ", len(adata.obs['layer_Spectral'].values))
    return adata

def affinity(adata, n):

    clf = AffinityPropagation(damping = 0.9, max_iter= 1000, preference=-5, random_state=0).fit(adata.obsm["embedding"])
    y_pred = clf.predict(adata.obsm["embedding"])
    adata.obs['layer_Affinity'] = y_pred
    print("Layer: ", adata.obs['layer_Affinity'].values, "Length: ", len(adata.obs['layer_Affinity'].values))
    return adata

def agglomerative(adata, n):

    clf = AgglomerativeClustering(n_clusters=n)
    y_pred = clf.fit_predict(adata.obsm["embedding"])
    adata.obs['layer_Agglomerative'] = y_pred
    print("Layer: ", adata.obs['layer_Agglomerative'].values, "Length: ", len(adata.obs['layer_Agglomerative'].values))
    return adata

def calculate(label, layer): #originally -> calculate(adata)

    """
    label = adata.obs['label'].values
    print("Label: ", adata.obs['label'].values, "Length: ", len(label))
    layer = adata.obs['layer'].values
    print("Layer: ", adata.obs['layer'].values, "Length: ", len(layer))
    """
    """
    # ------- note to self: why is it removing instances of cluster 0 in the labels?... ----------
    mask = []
    for item in range(len(label)):
        if label[item] != 0:
            mask.append(True)
        else:
            mask.append(False)
    print(mask)
    ARI = adjusted_rand_score(label[mask], layer[mask])
    print("Label Mask: ", label[mask], "Length: ", len(label[mask]))
    print("Layer Mask: ", layer[mask], "Length: ", len(layer[mask]))
    """
    ARI = adjusted_rand_score(label, layer)
    print(ARI)
    return ARI




if __name__ == '__main__':
    # args = parse_args()               # Add 8, 32, 128 to zdim
    scgnnsp_zdimList = ['256']#['3', '8', '16', '32', '64', '128', '256'] #['3','10', '16','32', '64', '128', '256']
    zdim_index = [f"zdim_{x}" for x in scgnnsp_zdimList]
    # scgnnsp_zdimList = ['3']
    scgnnsp_PEalphaList = ['5.0'] #['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'] #['0.1','0.2','0.3', '0.5', '1.0', '1.2', '1.5','2.0']
    PEalpha_col = [f"PEalpha_{x}" for x in scgnnsp_PEalphaList]
    # scgnnsp_PEalphaList = ['0.1']
    # sample_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674',
    #                '151675', '151676', '2-5', '2-8', '18-64', 'T4857']
    # sample_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674',
    #                '151675', '151676','2-5', '2-8', '18-64', 'T4857']
    # sample_list = ['2-5']

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

    samples = ['kidney085_XY04_21-0057']


    df = pd.DataFrame(index=pd.Index(zdim_index),
                        columns=PEalpha_col)
    df.to_csv('/result/kmeans.csv', index=True, header=True, mode = 'w')
    df.to_csv('/result/spectral.csv', index=True, header=True, mode = 'w')
    df.to_csv('/result/affinity.csv', index=True, header=True, mode = 'w')
    df.to_csv('/result/agglomerative.csv', index=True, header=True, mode = 'w')


    """
    df = pd.DataFrame(index=pd.Index(samples))
    for i in scgnnsp_zdimList:
        for j in scgnnsp_PEalphaList:
            column_name = 'PEalpha' + str(j) + '_zdim' + str(i)
            df[column_name] = None
    df.to_csv('/N/slate/mraina/RESEPT_EGNN/result/bugtest.csv', index=True, header=True, mode = 'w')
    """


    # core_num = cpu_count()
    pool = Pool(64)
    # for sample in sample_list:
        # pseudo_image_folder = output_path + "/" + sample + "/"
        # if not os.path.exists(pseudo_image_folder):
        #     os.makedirs(pseudo_image_folder)
    print("Start Main loop")
    #sample="057"
    for samp in samples:
        for scgnnsp_zdim in scgnnsp_zdimList:
            for scgnnsp_alpha in scgnnsp_PEalphaList:
                tmp=pool.apply_async(pseudo_images,(scgnnsp_zdim,scgnnsp_alpha,samp))
                    # pseudo_images(scgnnsp_zdim, scgnnsp_alpha, sample)
                tmp.get()
                #break
            #break
    pool.close()
    pool.join()
