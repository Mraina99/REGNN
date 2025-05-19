# Parse arguments
import argparse
import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
from REGNN_run.initialize_REGNN import initialize_REGNN
parser = argparse.ArgumentParser(description='Main program for REGNN')

# Data loading related
parser.add_argument('--load_dataset_dir', type=str, default='Data/', 
                    help='(str) Folder that stores all your datasets. For example, if your expression matrix is in /home/user/data/V10S14-085_XY04_21-0057/counts.csv, this should be set to /home/user/data/')
parser.add_argument('--load_dataset_name', type=str, default='V10S14-085_XY04_21-0057', 
                    help='(str) Folder that contains all the relevant input files. For example, if your expression matrix is in /home/user/data/V10S14-085_XY04_21-0057/counts.csv, this should be set to V10S14-085_XY04_21-0057')
parser.add_argument('--load_count_matrix', type=str, default='count.csv', 
                    help='Name of expression matrix file in data folder - default: count.csv')
parser.add_argument('--load_spatial', type=str, default='spa.csv', 
                    help='Name of spatial coordinates file in data folder - default: spa.csv')
parser.add_argument('--load_annotation_bool', type=bool, default=False, 
                    help='Name of spatial coordinates file in data folder - default: False')
parser.add_argument('--annotation_file', type=str, default='labels.csv', 
                    help='Name of spatial coordinates file in data folder - default: labels.csv')
parser.add_argument('--output_ARI', type=bool, default=False, 
                    help='Output ARI results for clustering embeddings -  default: False')

# Process related
parser.add_argument('--select_method', type=str, default='GAE', 
                    help='Use either REGNN GAE or SSL (GAE/SSL) -  default: GAE')
parser.add_argument('--n_clusters', type=int, default=4, 
                    help='Number of final clusters identified. If annotation file is used, n_clusters will automatically be number of classes in annotation file.')
parser.add_argument('--preprocess_top_gene_select', type=int, default=2000, 
                    help='Top genes kept after preprocessing')

parser.add_argument('--zdim', nargs="+", type=int, default=['3','8', '64', '128' ,'256'], 
                    help='list of zdim values to run')
parser.add_argument('--PEalpha', nargs="+", type=float, default=['1.0', '2.0', '3.0', '4.0', '4.5'], 
                    help='list of postional encoding alphas to run')

# Main program starts here
args = parser.parse_args()
print('\n> Starting REGNN ...')
initialize_REGNN(args)


print('\n> Program Finished! \n')