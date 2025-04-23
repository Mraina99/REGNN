#!/bin/sh
#SBATCH -J CCST
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=trstans@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH -A c01064

module load miniconda
conda activate CCST_2

python run_CCST.py --data_type nsc --data_name V10S14-085_XY01_20-0038 --lambda_I 0.8 --DGI 1 --load 0 --cluster 1 --PCA 1 --n_clusters 4 --draw_map 1 --diff_gene 0
