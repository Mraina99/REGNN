##############################################
# SEDR Benchmarking - Kidney 10x Visium
# Authors: Treyden Stansfield, Mauminah Raina
##############################################
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

import matplotlib.pyplot as plt
import seaborn as sns

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import SEDR

# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

# Load Datasets
dataList = [
'/N/u/trstans/Quartz/kidneydata2/kidneyV19S25-019_XY03_M61/'
]

# Make ARI file
with open('sedr_results_ARI.csv','a') as write:
	write.write('Sample,ARI\n')

for dataset in dataList:

	n_clusters = 4

	# Create anndata object
	adata = sc.read_csv(f"{dataset}count.csv")
	spatial = pd.read_csv(f"{dataset}spa.csv",sep=",",header=0,na_filter=False)
	adata.obsm['spatial'] = spatial
	df_meta = pd.read_csv(f'{dataset}labels.csv', sep=',')

	adata.obs['layer_guess'] = df_meta['class'].to_list()

	# Recommended preprocessing
	adata.layers['count'] = adata.X
	sc.pp.filter_genes(adata, min_cells=50)
	sc.pp.filter_genes(adata, min_counts=10)
	sc.pp.normalize_total(adata, target_sum=1e6)
	sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
	adata = adata[:, adata.var['highly_variable'] == True]
	sc.pp.scale(adata)

	from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
	adata_X = PCA(n_components=159, random_state=42).fit_transform(adata.X)
	adata.obsm['X_pca'] = adata_X

	# Neighborhood graph
	graph_dict = SEDR.graph_construction(adata, 12)
	print(graph_dict)

	# Train SEDR
	sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
	using_dec = True
	if using_dec:
    		sedr_net.train_with_dec(N=1)
	else:
    		sedr_net.train_without_dec(N=1)
	sedr_feat, _, _, _ = sedr_net.process()
	adata.obsm['SEDR'] = sedr_feat
	
	# Clustering
	SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')



	sub_adata = adata[~pd.isnull(adata.obs['layer_guess'])]
	

	ARI = metrics.cluster.adjusted_rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	AMI = metrics.cluster.adjusted_mutual_info_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	completeness = metrics.cluster.completeness_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	contingency = metrics.cluster.contingency_matrix(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	pair_confusion = metrics.cluster.pair_confusion_matrix(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	fowlkes = metrics.cluster.fowlkes_mallows_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	homogeneity_completeness = metrics.cluster.homogeneity_completeness_v_measure(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	homogeneity = metrics.cluster.homogeneity_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	normalized_mutual = metrics.cluster.normalized_mutual_info_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	rand = metrics.cluster.rand_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	v_measure = metrics.cluster.v_measure_score(sub_adata.obs['layer_guess'], sub_adata.obs['SEDR'])
	
	name = dataset.split('/')[6]
	print(name)
	#quit()
	with open('sedr_results.txt','a') as write:
		write.write(f'{name}\n')
		write.write(f'\tARI = {ARI}\n')
		write.write(f'\tAMI = {AMI}\n')
		write.write(f'\tcompleteness score = {completeness}\n')
		write.write(f'\tcontingency matrix = {contingency}\n')
		write.write(f'\tpair confusion = {pair_confusion}\n')
		write.write(f'\tfowlkes mallows score = {fowlkes}\n')
		write.write(f'\thomogeneity completeness = {homogeneity_completeness}\n')
		write.write(f'\thomogeneity score = {homogeneity}\n')
		write.write(f'\tnormalized mutual info = {normalized_mutual}\n')
		write.write(f'\trand score = {rand}\n')
		write.write(f'\tv measure score = {v_measure}\n')

	with open('sedr_results_ARI.csv','a') as write:
		write.write(f'{name},{ARI}\n')
	
	os.mkdir(f"sub_adata_results/{name}/")
        adata.write_csvs(f'sub_adata_results/{name}/')
