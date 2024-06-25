# Parse arguments
import argparse
import os,csv,re,json
import pandas as pd
import numpy as np
import scanpy as sc
import math
from REGNN_GAE.calculate_ARI import calc_ARI_GAE
from REGNN_SSL.calculate_ARI import calc_ARI_SSL
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
parser.add_argument('--load_annotation', type=str, default='labels.csv', 
                    help='Name of spatial coordinates file in data folder - default: labels.csv')
parser.add_argument('--select_method', type=str, default='GAE', 
                    help='Use either REGNN GAE or SSL (GAE/SSL) -  default: GAE')

# Preprocess related
parser.add_argument('--preprocess_cell_cutoff', type=float, default=0.9, 
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_gene_cutoff', type=float, default=0.9, 
                    help='Not needed if using benchmark')
parser.add_argument('--preprocess_top_gene_select', type=int, default=2000, 
                    help='Not needed if using benchmark')

args = parser.parse_args()


# Main program starts here
if args.select_method == "SSL":
    print('\n> Using REGNN_SSL ...')
    calc_ARI_SSL(args)
else: 
    print('\n> Using REGNN_GAE ...')
    calc_ARI_GAE(args)

print('\n> Program Finished! \n')