##############################################
# SpaGCN Benchmarking - Kidney 10x Visium
# Authors: Treyden Stansfield, Mauminah Raina
##############################################
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import SpaGCN as spg
import cv2
from scanpy import read_10x_h5
from sklearn.metrics.cluster import adjusted_rand_score

dataList = [
'/geode2/home/u100/trstans/Quartz/kidneydata2/kidneyV12N16-374_XY03_23-0112/'
]


for file in dataList:

	#Read original data and save it to h5ad
	from scanpy import read_10x_h5
	adata = sc.read_csv(file + "count.csv")
	spatial = pd.read_csv(file + "spa.csv",sep=",",header=0,na_filter=False)
	adata.obs["x1"]=spatial["tissue"]
	adata.obs["x2"]=spatial["row"]
	adata.obs["x3"]=spatial["col"]
	adata.obs["x4"]=spatial["imagerow"]
	adata.obs["x5"]=spatial["imagecol"]
	adata.obs["x_array"]=adata.obs["x2"]
	adata.obs["y_array"]=adata.obs["x3"]
	adata.obs["x_pixel"]=adata.obs["x4"]
	adata.obs["y_pixel"]=adata.obs["x5"]

	#Select captured samples
	adata=adata[adata.obs["x1"]==1]
	adata.var_names=[i.upper() for i in list(adata.var_names)]
	adata.var["genename"]=adata.var.index.astype("str")
	adata.write_h5ad(file + "SpacGCNdata.h5ad")

	#Read in gene expression and spatial location
	adata=sc.read(file + "SpacGCNdata.h5ad")

	#Read in hitology image
	#img=cv2.imread("~/kidneydata/kidney085_XY01_20-0038/SpacGCN_Toy_Data")


	#Set coordinates
	x_array=adata.obs["x_array"].tolist()
	y_array=adata.obs["y_array"].tolist()
	x_pixel=adata.obs["x_pixel"].tolist()
	y_pixel=adata.obs["y_pixel"].tolist()

	#Test coordinates on the image
	#img_new=img.copy()
	#for i in range(len(x_pixel)):
	#    x=x_pixel[i]
	#    y=y_pixel[i]
	#    img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

	#cv2.imwrite('/N/u/trstans/Quartz/kidneydata/kidney085_XY01_20-0038/151673_map.jpg', img_new)

	#Calculate adjacent matrix
	s=1
	b=49
	#adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
	adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
	np.savetxt(file + 'adj.csv', adj, delimiter=',')


	adata=sc.read(file + "SpacGCNdata.h5ad")
	adj=np.loadtxt(file + 'adj.csv', delimiter=',')
	adata.var_names_make_unique()
	spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
	spg.prefilter_specialgenes(adata)
	#Normalize and take log for UMI
	sc.pp.normalize_per_cell(adata)
	sc.pp.log1p(adata)


	p=0.5
	#Find the l value given p
	l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)



	n_clusters=4
	#Set seed
	r_seed=t_seed=n_seed=100
	#Search for suitable resolution
	res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)


	clf=spg.SpaGCN()
	clf.set_l(l)
	#Set seed
	random.seed(r_seed)
	torch.manual_seed(t_seed)
	np.random.seed(n_seed)
	#Run
	clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
	y_pred, prob=clf.predict()
	adata.obs["pred"]= y_pred
	adata.obs["pred"]=adata.obs["pred"].astype('category')
	#Do cluster refinement(optional)
	#shape="hexagon" for Visium data, "square" for ST data.
	adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
	refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
	adata.obs["refined_pred"]=refined_pred
	adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
	#Save results
	adata.write_h5ad(file + "results.h5ad")


	adata=sc.read(file + "results.h5ad")
	#adata.obs should contain two columns for x_pixel and y_pixel
	#Set colors used
	plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
	#Plot spatial domains
	domains="pred"
	num_celltype=len(adata.obs[domains].unique())
	adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
	ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
	ax.set_aspect('equal', 'box')
	ax.axes.invert_yaxis()
	plt.savefig(file + "pred.png", dpi=600)
	plt.close()

	#Plot refined spatial domains
	domains="refined_pred"
	num_celltype=len(adata.obs[domains].unique())
	adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
	ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
	ax.set_aspect('equal', 'box')
	ax.axes.invert_yaxis()
	plt.savefig(file + "refined_pred.png", dpi=600)
	plt.close()


	#This is additionally code added by Trey to get the ARI value
	SpacGCNadata=sc.read(file + 'results.h5ad')

	sample_adata = sc.read_csv(file + "count.csv")
	spatial = pd.read_csv(file + "spa.csv",sep=",",header=0,na_filter=False)
	adata.obs["x1"]=spatial["tissue"]
	adata.obs["x2"]=spatial["row"]
	adata.obs["x3"]=spatial["col"]
	adata.obs["x4"]=spatial["imagerow"]
	adata.obs["x5"]=spatial["imagecol"]
	adata.obs["x_array"]=adata.obs["x2"]
	adata.obs["y_array"]=adata.obs["x3"]
	adata.obs["x_pixel"]=adata.obs["x4"]
	adata.obs["y_pixel"]=adata.obs["x5"]


	labels_list = pd.read_csv(file + 'labels.csv')


	sample_adata.obs['class_labels'] = labels_list['class'].values

	
	print(adata.obs["pred"])
	print(sample_adata.obs['class_labels'])
	#print(labels_list)

	ARI = adjusted_rand_score(sample_adata.obs['class_labels'], SpacGCNadata.obs['pred'])
	print(ARI)
	with open('SpacGCN_ARI_list2.csv', 'a') as ariFile:
		ariFile.write(file.split("R")[0] + ',' + str(ARI) + '\n')
