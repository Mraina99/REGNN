import json
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import STAGATE_pyG

os.environ['R_HOME'] = '/geode2/home/u100/trstans/Quartz/.conda/envs/STAGATE/lib/R'
os.environ['R_USER'] = '/geode2/home/u100/trstans/Quartz/.conda/envs/STAGATE/lib/python3.10/site-packages/rpy2'

print(os.path.exists(os.environ['R_HOME']))
print(os.path.exists(os.environ['R_USER']))

# Initialize Data list
dataList = [
'/N/u/trstans/Quartz/kidneydata2/kidneyV10S14-085_XY01_20-0038/'
]

# Storing ARI
with open('STAGATE_results_ARI.csv','a') as write:
	write.write('Sample,ARI\n')

# Load data and run STAGATE
for dataset in dataList:

	# Set up anndata object
	adata = sc.read_csv(f"{dataset}count.csv")
	spatial = pd.read_csv(f"{dataset}spa.csv",sep=",",header=0,na_filter=False)

	imagerow = spatial['imagerow']
	imagecol = spatial['imagecol']

	d = {'imagerow':imagerow,'imagecol':imagecol}
	df = pd.DataFrame(data=d)


	adata.obsm['spatial'] = df

	adata.var_names_make_unique()
	adata.layers['count'] = adata.X
	sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)

	Ann_df = pd.read_csv(f'{dataset}labels.csv', sep=',')

	adata.obs['Ground Truth'] = Ann_df['class'].to_list()

	#with open('/N/u/trstans/Quartz/kidneydata2/kidneyV10S14-085_XY01_20-0038/sample.json', 'r') as file:
	#	spot_data = json.load(file)

	print(adata)

	#plt.rcParams["figure.figsize"] = (3, 3)
	#sc.pl.spatial(adata, spot_size=spot_data["spot_diameter_fullres"],img_key="hires", color=["Ground Truth"])

	# Construct Spatial Network
	STAGATE_pyG.Cal_Spatial_Net(adata, rad_cutoff=150)
	STAGATE_pyG.Stats_Spatial_Net(adata)

	# Run STAGATE
	adata = STAGATE_pyG.train_STAGATE(adata)

	sc.pp.neighbors(adata, use_rep='STAGATE')
	sc.tl.umap(adata)
	adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=4)

	obs_df = adata.obs.dropna()
	ARI = adjusted_rand_score(obs_df['Ground Truth'],obs_df['mclust'])
	print('Adjusted rand index = %.2f' %ARI)

	AMI = metrics.cluster.adjusted_mutual_info_score(obs_df['Ground Truth'],obs_df['mclust'])
	completeness = metrics.cluster.completeness_score(obs_df['Ground Truth'],obs_df['mclust'])
	contingency = metrics.cluster.contingency_matrix(obs_df['Ground Truth'],obs_df['mclust'])
	pair_confusion = metrics.cluster.pair_confusion_matrix(obs_df['Ground Truth'],obs_df['mclust'])
	fowlkes = metrics.cluster.fowlkes_mallows_score(obs_df['Ground Truth'],obs_df['mclust'])
	homogeneity_completeness = metrics.cluster.homogeneity_completeness_v_measure(obs_df['Ground Truth'],obs_df['mclust'])
	homogeneity = metrics.cluster.homogeneity_score(obs_df['Ground Truth'],obs_df['mclust'])
	normalized_mutual = metrics.cluster.normalized_mutual_info_score(obs_df['Ground Truth'],obs_df['mclust'])
	rand = metrics.cluster.rand_score(obs_df['Ground Truth'],obs_df['mclust'])
	v_measure = metrics.cluster.v_measure_score(obs_df['Ground Truth'],obs_df['mclust'])

	name = dataset.split('/')[6]
	print(name)
	#quit()
	with open('STAGATE_results.txt','a') as write:
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

	with open('STAGATE_results_ARI.csv','a') as write:
		write.write(f'{name},{ARI}\n')
	os.mkdir(f"sub_adata_results/{name}/")
	adata.write_csvs(f'sub_adata_results/{name}/')

