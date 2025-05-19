import time
import sys
import os
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
#import resource
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, OPTICS, MeanShift
from model import AE, VAE, PVAE, PAE
from util_function import *
from graph_function import *
from benchmark_util import *
from gae_embedding import GAEembedding, GAEembeddingMultiView, measure_clustering_results, test_clustering_benchmark_results
import torch.multiprocessing as mp
from sklearn.preprocessing import normalize
from models import DGI, logreg
from utils import process
import pdb
import aug

parser = argparse.ArgumentParser(description='Main entrance of scGNN')
parser.add_argument('--datasetName', type=str, default='481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                    help='For 10X: folder name of 10X dataset; For CSV: csv file name')
parser.add_argument('--datasetDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='Directory of dataset: default(/home/wangjue/biodata/scData/10x/6/)')

parser.add_argument('--batch-size', type=int, default=12800, metavar='N',
                    help='input batch size for training (default: 12800)')
parser.add_argument('--Regu-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train in Feature Autoencoder (default: 500)')
parser.add_argument('--EM-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train in iteration EM (default: 200)')
parser.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                    help='number of iteration in total EM iteration (default: 10)')
parser.add_argument('--quickmode', action='store_true', default=False,
                    help='whether use quickmode, skip Cluster Autoencoder (default: no quickmode)')
parser.add_argument('--cluster-epochs', type=int, default=200, metavar='N',
                    help='number of epochs in Cluster Autoencoder training (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable GPU training. If you only have CPU, add --no-cuda in the command line')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regulized-type', type=str, default='noregu',
                    help='regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--reduction', type=str, default='sum',
                    help='reduction type: mean/sum, default(sum)')
parser.add_argument('--model', type=str, default='PAE',
                    help='VAE/AE/PVAE/PAE (default: PAE),AE for scGNN')
parser.add_argument('--gammaPara', type=float, default=0.1,
                    help='regulized intensity (default: 0.1)')
parser.add_argument('--alphaRegularizePara', type=float, default=0.9,
                    help='regulized parameter (default: 0.9)')

# Build cell graph
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation/cityblock (default: euclidean)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThread',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread/spatialGrid/Lattice (default: KNNgraphStatsSingleThread)')
parser.add_argument('--pruneTag', type=str, default='NA',
                    help='prune tag, NA/STD/GridEx/GridEx2/../GridEx18 (default: NA)')
parser.add_argument('--adjtype', type=str, default='unweighted',
                    help='adjtype (default: unweighted) otherwise: weighted')

# Debug related
parser.add_argument('--precisionModel', type=str, default='Float',
                    help='Single Precision/Double precision: Float/Double (default:Float)')
parser.add_argument('--coresUsage', type=str, default='1',
                    help='how many cores used: all/1/... (default:1)')
parser.add_argument('--outputDir', type=str, default='outputDir/',
                    help='save npy results in directory')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--saveinternal', action='store_true', default=False,
                    help='whether save internal interation results or not')
parser.add_argument('--debugMode', type=str, default='noDebug',
                    help='savePrune/loadPrune for extremely huge data in debug (default: noDebug)')
parser.add_argument('--nonsparseMode', action='store_true', default=False,
                    help='SparseMode for running for huge dataset')

# LTMG related
parser.add_argument('--LTMGDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--ltmgExpressionFile', type=str, default='Use_expression.csv',
                    help='expression File after ltmg in csv')
parser.add_argument('--ltmgFile', type=str, default='LTMG_sparse.mtx',
                    help='expression File in csv. (default:LTMG_sparse.mtx for sparse mode/ ltmg.csv for nonsparse mode) ')

# Clustering related
parser.add_argument('--embedding', type=str, default='GAE',
                    help='Embedding Type: SSL/GAE')
parser.add_argument('--useGAEembedding', action='store_true', default=False,
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False,
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--n-clusters', default=20, type=int,
                    help='number of clusters if predifined for KMeans/Birch ')
parser.add_argument('--clustering-method', type=str, default='LouvainK',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)')
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')

# imputation related
parser.add_argument('--EMregulized-type', type=str, default='Celltype',
                    help='regulized type (default: noregu) in EM, otherwise: noregu/Graph/GraphR/Celltype')
parser.add_argument('--gammaImputePara', type=float, default=0.0,
                    help='regulized parameter (default: 0.0)')
parser.add_argument('--graphImputePara', type=float, default=0.3,
                    help='graph parameter (default: 0.3)')
parser.add_argument('--celltypeImputePara', type=float, default=0.1,
                    help='celltype parameter (default: 0.1)')
parser.add_argument('--L1Para', type=float, default=1.0,
                    help='L1 regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.0,
                    help='L2 regulized parameter (default: 0.001)')
parser.add_argument('--EMreguTag', action='store_true', default=False,
                    help='whether regu in EM process')
parser.add_argument('--sparseImputation', type=str, default='nonsparse',
                    help='whether use sparse in imputation: sparse/nonsparse (default: nonsparse)')

# spatial related
parser.add_argument('--useSpatial', action='store_true', default=False,
                    help='whether use spatial information')
parser.add_argument('--spatialFile', type=str, default='coords_array.npy',
                    help='file contains spatial information')

# dealing with zeros in imputation results
parser.add_argument('--zerofillFlag', action='store_true', default=False,
                    help='fill zero or not before EM process (default: False)')
parser.add_argument('--noPostprocessingTag', action='store_false', default=True,
                    help='whether postprocess imputated results, default: (True)')
parser.add_argument('--postThreshold', type=float, default=0.01,
                    help='Threshold to force expression as 0, default:(0.01)')

# PVAE/PAE related (scGNNsp)
parser.add_argument('--zdim', type=int, default=10, help='Dimension of latent variable in PVAE')
parser.add_argument('--beta', default=1.0, help='Choice of beta schedule or a constant for KLD weight (default: %(default)s)')
parser.add_argument('--beta-control', type=float, help='KL-Controlled VAE gamma. Beta is KL target. (default: %(default)s)')

parser.add_argument('--qlayers', type=int,
                    default=5, help='Number of hidden layers of encoder (default: %(default)s), original: 3')
parser.add_argument('--qdim', type=int, default=512,
                    help='Number of nodes in hidden layers of encoder (default: %(default)s), original: 256')
parser.add_argument('--encode-mode', default='resid', choices=('conv', 'resid',
                                                               'mlp', 'tilt', 'cluster'), help='Type of encoder network (default: %(default)s)')
parser.add_argument('--players', type=int, default=5,
                    help='Number of hidden layers of decoder(default: %(default)s), original: 3')
parser.add_argument('--pdim', type=int, default=512,
                    help='Number of nodes in hidden layers of decoder (default: %(default)s), original: 256')
parser.add_argument('--decode-mode', choices=('resid', 'cluster'), default='resid', help='Type of decoder network (default: %(default)s)')
parser.add_argument('--pe-type', choices=('geom_ft', 'geom_full', 'geom_lowf', 'geom_nohighf', 'linear_lowf',
                                          'none','dummy','rope'), default='geom_lowf', help='Type of positional encoding (default: %(default)s)')
parser.add_argument('--pe-dim', type=int,
                    help='Num features in positional encoding (default: image D)')
parser.add_argument('--domain', choices=('hartley', 'fourier'), default='fourier',
                    help='Decoder representation domain (default: %(default)s)')
parser.add_argument('--activation', choices=('relu', 'leaky_relu'),
                    default='relu', help='Activation (default: %(default)s)')
parser.add_argument('--PEtypeOp', choices=('concat', 'add', 'multi'),
                    default='add', help='Positional Encoding type (default: %(default)s)')
parser.add_argument('--PEalpha', type=float, default=1.0,
                    help='Positional Encoding alpha (default: %(default)s)')
parser.add_argument('--bypassAE', action='store_true', default=False,
                    help='whether bypass Feature AE (default: False)')

# Converge related
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge-type', type=str, default='celltype',
                    help='type of converge condition: celltype/graph/both/either (default: celltype) ')
parser.add_argument('--converge-graphratio', type=float, default=0.01,
                    help='converge condition: ratio of graph ratio change in EM iteration (default: 0.01), 0-1')
parser.add_argument('--converge-celltyperatio', type=float, default=0.99,
                    help='converge condition: ratio of cell type change in EM iteration (default: 0.99), 0-1')

# GAE related
parser.add_argument('--GAEmodel', type=str,
                    default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16,
                    help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001,
                    help='Initial learning rate for regularization.')

# New Spatail Exploration
parser.add_argument('--MVstrategy', choices=('SandC','C2S','NA','hybrid'), default='NA', help='Type of MultiView integration (default: %(default)s)') #Original useMultiView
parser.add_argument('--MVop', choices=('intersect', 'union', 'concat'), default='intersect', help='Type of MultiView integration integration/union (default: %(default)s)')
parser.add_argument('--MVGrid', choices=('ori', 'ex'), default='ori', help='Type of spatial: ori/ex (default: %(default)s)')
parser.add_argument('--Sk', type=str, default='160', help='Spatial K (default: %(default)s)')
parser.add_argument('--SpruneTag', type=str, default='GridEx19', help='Spatial Prunetype (default: %(default)s)')


# contractive learning
# parser.add_argument('--dataset',          type=str,           default="citeseer",                help='data')
parser.add_argument('--aug_type',         type=str,           default="edge",                help='augmentation type: edge/node/mask/subgraph')
parser.add_argument('--drop_percent',     type=float,         default=0.1,               help='drop percent')
parser.add_argument('--gpu',              type=int,           default=0,                 help='gpu')
# parser.add_argument('--save_name',        type=str,           default='try.pkl',                help='save ckpt name')
parser.add_argument('--hidunits',         type=int,           default=3,                 help='num of hid units')
parser.add_argument('--SSLpatience',         type=int,           default=2000,                 help='patience')
parser.add_argument('--SSLepochs',         type=int,           default=10000,                 help='SSL epochs')
parser.add_argument('--SSLlr',         type=float,           default=0.001,                 help='lr in SSL')

# To identify Spatial Variable Genes
parser.add_argument('--svg', action='store_true', default=False,
                    help='whether to infer spatial variable gene')
parser.add_argument('--svginfertype', choices=('aug', 'noaug'),
                    default='aug', help='svg infer type (default: %(default)s)')
parser.add_argument('--graphBatch',         type=int,           default=40,                 help='graph batch size')

# Graph
# GNN settings
parser.add_argument('--Gmodel', type=str, default='DGCNN')
parser.add_argument('--Gsortpool_k', type=float, default=0.6)
parser.add_argument('--Gnum_layers', type=int, default=3)
parser.add_argument('--Ghidden_channels', type=int, default=32)
parser.add_argument('--Gbatch_size', type=int, default=32)
# Subgraph extraction settings
parser.add_argument('--Gnum_hops', type=int, default=1)
parser.add_argument('--Gratio_per_hop', type=float, default=1.0)
parser.add_argument('--Gmax_nodes_per_hop', type=int, default=None)
parser.add_argument('--Gnode_label', type=str, default='drnl', 
                    help="which specific labeling trick to use")
parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")
# Training settings
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_percent', type=float, default=100)
parser.add_argument('--val_percent', type=float, default=100)
parser.add_argument('--test_percent', type=float, default=100)
parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
parser.add_argument('--dynamic_val', action='store_true')
parser.add_argument('--dynamic_test', action='store_true')
parser.add_argument('--num_workers', type=int, default=16, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
parser.add_argument('--train_node_embedding', action='store_true', 
                    help="also train free-parameter node embeddings together with GNN")
parser.add_argument('--pretrained_node_embedding', type=str, default=None, 
                    help="load pretrained node embeddings as additional node features")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.sparseMode = not args.nonsparseMode
"""
seed = args.seed
#random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
"""

aug_type = args.aug_type
drop_percent = args.drop_percent
hid_units = args.hidunits
# old
# save_name = args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_'+args.MVop+'_'+args.Sk+'_'+args.SpruneTag+'_'+args.aug_type+'_'+str(args.drop_percent)+'_'+str(hid_units)
#
modelname = args.model
if args.bypassAE:
    modelname = 'ori'
save_name = args.outputDir+args.datasetName+'_'+modelname+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_'+args.aug_type+'_'+str(args.drop_percent)+'_'+str(hid_units)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.sparseMode = not args.nonsparseMode

# TODO
# As we have lots of parameters, should check args
checkargs(args)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device:'+str(device))

if not args.coresUsage == 'all':
    torch.set_num_threads(int(args.coresUsage))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# print(args)
start_time = time.time()

# load scRNA in csv
print('---0:00:00---scRNA starts loading.')
data, genelist, celllist = loadscExpression(
    args.datasetDir+args.datasetName+'/'+args.ltmgExpressionFile, sparseMode=args.sparseMode)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---scRNA has been successfully loaded')

scData = scDataset(data)
#print("data: ", data)
train_loader = DataLoader(
    scData, batch_size=args.batch_size, shuffle=False, **kwargs)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---TrainLoader has been successfully prepared.')

# load LTMG in sparse version
if not args.regulized_type == 'noregu':
    print('Start loading LTMG in sparse coding.')
    regulationMatrix = readLTMG(
        args.LTMGDir+args.datasetName+'/', args.ltmgFile)
    regulationMatrix = torch.from_numpy(regulationMatrix)
    if args.precisionModel == 'Double':
        regulationMatrix = regulationMatrix.type(torch.DoubleTensor)
    elif args.precisionModel == 'Float':
        regulationMatrix = regulationMatrix.type(torch.FloatTensor)
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---LTMG has been successfully prepared.')
else:
    regulationMatrix = None

# load spatial information
if args.useSpatial:
    spatialMatrix = readSpatial(args.datasetDir+args.datasetName+'/'+args.spatialFile)
    spatialMatrix = preprocessSpatial(spatialMatrix)
    spatialMatrix = torch.from_numpy(spatialMatrix)
    spatialMatrix = spatialMatrix.type(torch.FloatTensor)
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---Spatial information has been successfully loaded.')
    #print("Spatial Info: ", spatialMatrix.shape)
else:
    spatialMatrix = None

# Original
if args.model == 'VAE':
    model = VAE(dim=scData.features.shape[1]).to(device)
elif args.model == 'AE':
    model = AE(dim=scData.features.shape[1]).to(device)
elif args.model == 'PVAE':
    # Results are very bad
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    model = PVAE(D=args.zdim, qlayers=args.qlayers, qdim=args.qdim, players=args.players, pdim=args.pdim,
                 in_dim=scData.features.shape[1], zdim=args.zdim, outdim=scData.features.shape[1], encode_mode=args.encode_mode, decode_mode=args.decode_mode,
                 enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain,
                 activation=activation, petype=args.PEtypeOp, pe_alpha=args.PEalpha).to(device)
elif args.model == 'PAE':
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[args.activation]
    #print("in_dim: ", scData.features.shape[1], "pdim: ", args.pdim)
    model = PAE(D=args.zdim, qlayers=args.qlayers, qdim=args.qdim, players=args.players, pdim=args.pdim,
                 in_dim=scData.features.shape[1], zdim=args.zdim, outdim=scData.features.shape[1], encode_mode='cluster', decode_mode='cluster',
                 enc_type=args.pe_type, enc_dim=args.pe_dim, domain=args.domain, #enc_type=args.pe_type
                 activation=activation, petype=args.PEtypeOp, pe_alpha=args.PEalpha).to(device)
#print(model)
"""
print("model parameters: ")
for param in model.parameters():
    print(param)
"""

if args.precisionModel == 'Double':
    model = model.double()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---Pytorch model ready.')


def train(epoch, train_loader=train_loader, EMFlag=False, taskType='celltype', sparseImputation='nonsparse'):
    '''
    EMFlag indicates whether in EM processes. 
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
        taskType: celltype or imputation
    '''
    model.train()
    train_loss = 0
    for batch_idx, (data, dataindex) in enumerate(train_loader):
        if args.precisionModel == 'Double':
            data = data.type(torch.DoubleTensor)
        elif args.precisionModel == 'Float':
            data = data.type(torch.FloatTensor)
        data = data.to(device)

        # bypass feature autoEncoder, use original input
        if args.bypassAE:
            recon_batch = data
            z = data
        
        # Classical usage with Feature autoEncoder
        else:
            # LTMG
            if not args.regulized_type == 'noregu':
                regulationMatrixBatch = regulationMatrix[dataindex, :]
                regulationMatrixBatch = regulationMatrixBatch.to(device)
            else:
                regulationMatrixBatch = None

            # spatial
            if args.useSpatial:
                spatialMatrixBatch = spatialMatrix[dataindex, :]
                spatialMatrixBatch = spatialMatrixBatch.to(device)
            else:
                spatialMatrixBatch = None

            if taskType == 'imputation':
                if sparseImputation == 'nonsparse':
                    celltypesampleBatch = celltypesample[dataindex,
                                                        :][:, dataindex]
                    adjsampleBatch = adjsample[dataindex, :][:, dataindex]
                elif sparseImputation == 'sparse':
                    celltypesampleBatch = generateCelltypeRegu(
                        listResult[dataindex])
                    celltypesampleBatch = torch.from_numpy(celltypesampleBatch)
                    if args.precisionModel == 'Float':
                        celltypesampleBatch = celltypesampleBatch.float()
                    elif args.precisionModel == 'Double':
                        celltypesampleBatch = celltypesampleBatch.type(
                            torch.DoubleTensor)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('celltype Mem consumption: '+str(mem))

                    adjsampleBatch = adj[dataindex, :][:, dataindex]
                    adjsampleBatch = sp.csr_matrix.todense(adjsampleBatch)
                    adjsampleBatch = torch.from_numpy(adjsampleBatch)
                    if args.precisionModel == 'Float':
                        adjsampleBatch = adjsampleBatch.float()
                    elif args.precisionModel == 'Double':
                        adjsampleBatch = adjsampleBatch.type(torch.DoubleTensor)
                    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    # print('adj Mem consumption: '+str(mem))

            optimizer.zero_grad()
            if args.model == 'VAE':
                recon_batch, mu, logvar, z = model(data)
                if taskType == 'celltype':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, 
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, 
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)

            elif args.model == 'AE':
                recon_batch, z = model(data)
                mu_dummy = ''
                logvar_dummy = ''
                if taskType == 'celltype':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=args.gammaPara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=args.gammaPara, 
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)

            # positional VAE
            elif args.model == 'PVAE':

                B = data.size(0)
                # encode
                z_mu, z_logvar = model.encode(data)
                z = model.reparameterize(z_mu, z_logvar)
                # decode
                recon_batch = model(spatialMatrixBatch, z).view(B, -1)

                # Directly use Loss
                # loss = loss_function_position(z_mu, z_logvar, data, recon_batch, args.beta)

                if taskType == 'celltype':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, 
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, 
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)


            # positional AE
            elif args.model == 'PAE':

                B = data.size(0)

                # encode
                z = model.encode(data)

                # decode
                recon_batch = model(spatialMatrixBatch, z).view(B, -1)

                mu_dummy = ''
                logvar_dummy = ''

                # Directly use Loss
                # loss = loss_function_position_AE(data, recon_batch, args.beta)

                if taskType == 'celltype':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=args.gammaPara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=args.gammaPara, 
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                elif taskType == 'imputation':
                    if EMFlag and (not args.EMreguTag):
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)
                    else:
                        loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                            regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)

            # L1 and L2 regularization
            # 0.0 for no regularization
            l1 = 0.0
            l2 = 0.0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
                l2 = l2 + p.pow(2).sum()
            loss = loss + args.L1Para * l1 + args.L2Para * l2

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        # for batch
        if batch_idx == 0:
            recon_batch_all = recon_batch
            data_all = data
            z_all = z
        else:
            recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
            data_all = torch.cat((data_all, data), 0)
            z_all = torch.cat((z_all, z), 0)

    if not args.bypassAE:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return recon_batch_all, data_all, z_all


def process_edges(edgeList):
    result=[]
    for i in edgeList:
        if i[2]>0:
            result.append([i[0],i[1]])
    result=np.array(result)
    edges=[torch.tensor(result[:,0],dtype=torch.int64),
           torch.tensor(result[:,1],dtype=torch.int64)]

    return edges

if __name__ == "__main__":
    start_time = time.time()
    adjsample = None
    celltypesample = None
    # If not exist, then create the outputDir
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    # outParaTag = str(args.gammaImputePara)+'-'+str(args.graphImputePara)+'-'+str(args.celltypeImputePara)
    ptfileStart = args.outputDir+args.datasetName+'_EMtrainingStart.pt'
    # ptfile      = args.outputDir+args.datasetName+'_EMtraining.pt'

    # Debug
    if args.debugMode == 'savePrune' or args.debugMode == 'noDebug':
        # store parameter
        stateStart = {
            # 'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(stateStart, ptfileStart)
        print('Start training...')
        epochs = args.Regu_epochs + 1
        # for bypass feature AE, only run onces
        if args.bypassAE:
            epochs = 2
        # 这里的epochs改成10
        for epoch in range(1, epochs):
            recon, original, z = train(epoch, EMFlag=False)
            #print("Outloop = Recon: ", recon.shape)  
            #print("Outloop = original: ", original.shape)  
            #print("Outloop = z: ", z.shape)  

        zOut = z.detach().cpu().numpy()
        #print('zOut ready at ' + str(time.time()-start_time))
        #print("ZOUT: ", zOut)
        ptstatus = model.state_dict()

        # Store reconOri for imputation
        reconOri = recon.clone()
        reconOri = reconOri.detach().cpu().numpy()

        # Step 1. Inferring celltype

        # Here para = 'euclidean:10'
        # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k))
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'---Start Prune')
        if args.adjtype == 'unweighted': 
            # Hybird of spatial and correlation graph   
            if args.MVstrategy == 'hybrid':
                print('Using hybrid MVstrategy')
                adjBasicS, edgeListBasicS = generateAdj(zOut, graphType='spatialGrid', para='euclidean:8:Grid', adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)
                print(len(edgeListBasicS))
                adjBasicSe, edgeListBasicSe = generateAdj(zOut, graphType='spatialGrid', para='euclidean'+':'+args.Sk+':'+args.SpruneTag, adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)
                print(len(edgeListBasicSe))
                adjBasicC, edgeListBasicC = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k)+':'+args.pruneTag, adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix) 
                print(len(edgeListBasicC))
                adj, edgeList = generateMVedgeList(edgeListBasicC, edgeListBasicSe, zOut.shape[0], MVop='intersect')
                print(len(edgeList))   
                # adj, edgeList = generateMVedgeList(edgeListBasicS, edgeList, zOut.shape[0], MVop='union')
                adj, edgeList = generateMVedgeList(edgeListBasicS, edgeList, zOut.shape[0], MVop='concat')
                print(len(edgeList))                 

            # Original of spatial Graph and correlation Graph                    
            elif args.MVstrategy == 'NA':
                print('Using original MVstrategy')
                adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k)+':'+args.pruneTag, adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)
                #print(len(edgeList))
            
            # MultiView
            elif args.MVstrategy == 'C2S' or args.MVstrategy == 'SandC':
                print('Using C2S/SandC MVstrategy')
                # Generate Source and Target graph
                if args.MVGrid == 'ori': # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
                    adjTarget, edgeListT = generateAdj(zOut, graphType='spatialGrid', para='euclidean:8:Grid', adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)
                    
                elif args.MVGrid == 'ex': # Prune: only use nearest/second nearest neighbor as exact grid: 12 in cityblock, 16 in eculidean
                    adjTarget, edgeListT = generateAdj(zOut, graphType='spatialGrid', para='euclidean:16:GridEx', adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)
                adjSource, edgeListS = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k)+':'+args.pruneTag, adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)    
                
                # For edgeList
                if args.MVstrategy == 'C2S':
                    edgeList = edgeListS
                elif args.MVstrategy == 'SandC':
                    adj, edgeList = generateMVedgeList(edgeListT, edgeListS, zOut.shape[0], MVop=args.MVop)
            
            #print("Adjaceny Matrix : ", adj)
            #print("Adjaceny Matrix shape : ", adj.shape)
                                
        elif args.adjtype == 'weighted':
            adj, edgeList = generateAdjWeighted(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k), adjTag=(args.useGAEembedding or args.useBothembedding), spatialMatrix = spatialMatrix)
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Prune Finished')
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        if args.debugMode == 'savePrune':
            # Add protocol=4 for serizalize object larger than 4GiB
            # with open('edgeListFile', 'wb') as edgeListFile:
            #     pkl.dump(edgeList, edgeListFile, protocol=4)

            # with open('adjFile', 'wb') as adjFile:
            #     pkl.dump(adj, adjFile, protocol=4)

            with open(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_zOutFile', 'wb') as zOutFile:
                pkl.dump(zOut, zOutFile, protocol=4)

            # with open('reconFile', 'wb') as reconFile:
            #     pkl.dump(recon, reconFile, protocol=4)

            # with open('originalFile', 'wb') as originalFile:
            #     pkl.dump(original, originalFile, protocol=4)

            # sys.exit(0)

    if args.debugMode == 'loadPrune':

        with open('edgeListFile', 'rb') as edgeListFile:
            edgeList = pkl.load(edgeListFile)

        with open('adjFile', 'rb') as adjFile:
            adj = pkl.load(adjFile)

        with open('zOutFile', 'rb') as zOutFile:
            zOut = pkl.load(zOutFile)

        with open('reconFile', 'rb') as reconFile:
            recon = pkl.load(reconFile)

        with open('originalFile', 'rb') as originalFile:
            original = pkl.load(originalFile)



    if args.embedding=='GAE' or args.embedding=='SSL' or args.embedding=='Both':
        # Whether use GAE embedding
        if args.useGAEembedding or args.useBothembedding:
            zDiscret = zOut > np.mean(zOut, axis=0) # normalize(zOut, axis=0) 
            zDiscret = 1.0*zDiscret
            if args.useGAEembedding:
                edges=process_edges(edgeList)
                if args.MVstrategy == 'C2S':
                    zOut = GAEembeddingMultiView(zDiscret, adjSource, adjTarget, args)
                elif args.MVstrategy == 'NA' or args.MVstrategy =='SandC' or args.MVstrategy =='hybrid':
                    zOut = GAEembedding(zDiscret, adj, spatialMatrix,edges,args)
                print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                            start_time)))+"---GAE embedding finished")
                #print("zOut AFTER GAE: ", zOut.shape)
                #print("zOut AFTER GAE: ", zOut)

            elif args.useBothembedding:
                # It looks like does not work
                if args.MVstrategy == 'C2S':
                    zEmbedding = GAEembeddingMultiView(zDiscret, adjSource, adjTarget, args)
                elif args.MVstrategy == 'NA' or args.MVstrategy =='SandC' or args.MVstrategy =='hybrid':
                    zEmbedding = GAEembedding(zDiscret, adj, args)
                zOut = np.concatenate((zOut, zEmbedding), axis=1)
        
        # Code block to handle when embedding includes NaN
        if (zOut != zOut).any() == True: 
            emblist = []
            for i in range(zOut.shape[1]):
                emblist.append('embedding'+str(i))
            embedding_df = pd.DataFrame(zOut, index=celllist, columns=emblist)
            # embedding_df.to_csv(args.outputDir+args.datasetName+'_embedding.csv')
            embedding_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_'+args.MVop+'_'+args.Sk+'_'+args.SpruneTag+'_embedding.csv')
            sys.exit("Embedding contains NaNs - 'Calculate_ARI.py' failsafe")
    # GraphCL in SSL
    if args.embedding=='SSL':
        print("Using SSL")
        # training params
        batch_size = 1
        # nb_epochs = 10000
        nb_epochs = args.SSLepochs
        patience = args.SSLpatience
        # patience = 2
        lr = args.SSLlr
        l2_coef = 0.0
        drop_prob = 0.0
        #hid_units = 16
        # Now sparse is False
        # TODO: will use sparse to save time and memory
        # sparse = True
        sparse = False

        #svg
        graph_batch_size=args.graphBatch

        nonlinearity = 'prelu' # special name to separate parameters
        # adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
        # features, _ = process.preprocess_features(features)

        features = zOut

        nb_nodes = features.shape[0]  # node number
        ft_size = features.shape[1]   # node features dim
        # nb_classes = labels.shape[1]  # classes = 6

        features = torch.FloatTensor(features[np.newaxis])

        # detecting svg:
        if args.svg:
            #temp Batch is using all in the Batch
            # features (1,nodes,att) -> (att,nodes,1)
            # features = torch.einsum('ijk->kji',features)
            ft_size = 1
            hid_units = 1

        '''
        ------------------------------------------------------------
        edge node mask subgraph
        ------------------------------------------------------------
        '''
        print("Begin Aug:[{}]".format(args.aug_type))
        if args.aug_type == 'edge':
            if args.svg:
                features = torch.einsum('ijk->kji',features)

            aug_features1 = features
            aug_features2 = features

            aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
            aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent) # random drop edges
            
        elif args.aug_type == 'node':
            
            aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
            aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)

            if args.svg:
                features = torch.einsum('ijk->kji',features)
                aug_features1 = torch.einsum('ijk->kji',aug_features1)
                aug_features2 = torch.einsum('ijk->kji',aug_features2)
            
        elif args.aug_type == 'subgraph':
            
            aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
            aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)

            if args.svg:
                features = torch.einsum('ijk->kji',features)
                aug_features1 = torch.einsum('ijk->kji',aug_features1)
                aug_features2 = torch.einsum('ijk->kji',aug_features2)

        # if svg, don't use mask
        elif args.aug_type == 'mask':

            aug_features1 = aug.aug_random_mask(features,  drop_percent=drop_percent)
            aug_features2 = aug.aug_random_mask(features,  drop_percent=drop_percent)
            
            aug_adj1 = adj
            aug_adj2 = adj

        else:
            assert False

        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'---Finish aug')
        '''
        ------------------------------------------------------------
        '''

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
        aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
            sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
            aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
            aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()


        '''
        ------------------------------------------------------------
        mask
        ------------------------------------------------------------
        '''

        '''
        ------------------------------------------------------------
        '''
        if not sparse:
            if args.svg:
                adj = torch.FloatTensor(adj)
                aug_adj1 = torch.FloatTensor(aug_adj1)
                aug_adj2 = torch.FloatTensor(aug_adj2)
                # adj = torch.FloatTensor(np.repeat(adj[np.newaxis],features.shape[0],axis=0))
                # aug_adj1 = torch.FloatTensor(np.repeat(aug_adj1[np.newaxis],features.shape[0],axis=0))
                # aug_adj2 = torch.FloatTensor(np.repeat(aug_adj2[np.newaxis],features.shape[0],axis=0))

            else:
                adj = torch.FloatTensor(adj[np.newaxis])
                aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
                aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])


        # labels = torch.FloatTensor(labels[np.newaxis])
        # idx_train = torch.LongTensor(idx_train)
        # idx_val = torch.LongTensor(idx_val)
        # idx_test = torch.LongTensor(idx_test)

        model = DGI.DGI(ft_size, hid_units, nonlinearity)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        if torch.cuda.is_available():
            print('Using CUDA')
            model.cuda()
            features = features.cuda()
            aug_features1 = aug_features1.cuda()
            aug_features2 = aug_features2.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
                sp_aug_adj1 = sp_aug_adj1.cuda()
                sp_aug_adj2 = sp_aug_adj2.cuda()
            else:
                adj = adj.cuda()
                aug_adj1 = aug_adj1.cuda()
                aug_adj2 = aug_adj2.cuda()

            # labels = labels.cuda()
            # idx_train = idx_train.cuda()
            # idx_val = idx_val.cuda()
            # idx_test = idx_test.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        graphEmb_best = ''

        #SVG
        if args.svg:
            for epoch in range(nb_epochs):

                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]

                lbl_1 = torch.ones(graph_batch_size, nb_nodes)
                lbl_2 = torch.zeros(graph_batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.cuda()
                    lbl = lbl.cuda()
                                    
                graph_step = int(features.shape[0]/graph_batch_size)
                loss_t = 0
                graphEmb_tmp = np.zeros((features.shape[0],features.shape[1],hid_units))
                for batch_i in np.arange(graph_step):
                    model.train()
                    optimiser.zero_grad()

                    logits,graphEmb = model(features[batch_i*graph_batch_size:(batch_i+1)*graph_batch_size,:,:], 
                        shuf_fts[batch_i*graph_batch_size:(batch_i+1)*graph_batch_size,:,:], 
                        aug_features1[batch_i*graph_batch_size:(batch_i+1)*graph_batch_size,:,:], 
                        aug_features2[batch_i*graph_batch_size:(batch_i+1)*graph_batch_size,:,:],
                        sp_adj if sparse else adj.repeat(graph_batch_size,1,1), 
                        sp_aug_adj1 if sparse else aug_adj1.repeat(graph_batch_size,1,1),
                        sp_aug_adj2 if sparse else aug_adj2.repeat(graph_batch_size,1,1),  
                        sparse, None, None, None, aug_type=aug_type)

                    graphEmb_tmp[batch_i*graph_batch_size:(batch_i+1)*graph_batch_size,:,:] = graphEmb.detach().cpu().numpy()
                    loss = b_xent(logits, lbl)
                    loss.backward()
                    optimiser.step()
                    loss_t = loss_t + loss.item()

                    # print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'batch:'+str(batch_i))
                    
                print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'---epoch'+str(epoch)+'\tLoss:[{:.4f}]'.format(loss_t/features.shape[0]))

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), save_name+'.pkl')
                    graphEmb_best = graphEmb_tmp
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    break

            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load(save_name+'.pkl'))

        # Original
        else:
            for epoch in range(nb_epochs):

                model.train()
                optimiser.zero_grad()

                idx = np.random.permutation(nb_nodes)
                shuf_fts = features[:, idx, :]

                lbl_1 = torch.ones(batch_size, nb_nodes)
                lbl_2 = torch.zeros(batch_size, nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1)

                if torch.cuda.is_available():
                    shuf_fts = shuf_fts.cuda()
                    lbl = lbl.cuda()

                logits,graphEmb = model(features, shuf_fts, aug_features1, aug_features2,
                            sp_adj if sparse else adj, 
                            sp_aug_adj1 if sparse else aug_adj1,
                            sp_aug_adj2 if sparse else aug_adj2,  
                            sparse, None, None, None, aug_type=aug_type) 

                loss = b_xent(logits, lbl)
                print('Loss:[{:.4f}]'.format(loss.item()))

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), save_name+'.pkl')
                    graphEmb_best = graphEmb
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    break

                loss.backward()
                optimiser.step()

            print('Loading {}th epoch'.format(best_t))
            model.load_state_dict(torch.load(save_name+'.pkl'))

        if args.svg:
            print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---Training finished, save the learned results.')
            np.save(save_name+'_graph.npy',graphEmb_best)
            print('Done, Save')
            exit()
        
        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
        zOut = embeds[0,:]

        #print("zOut after SSL: ", zOut)
        #print("zOut AFTER SSL: ", zOut.shape)
        
        if torch.cuda.is_available():
            zOut = z.detach().cpu().numpy()

        zOut = zOut.detach().cpu().numpy()
        #print("zOut after SSL: ", zOut)
        #print("zOut AFTER SSL: ", zOut.shape)
            


 ######################################  
            

    # For iteration studies
    G0 = nx.Graph()
    G0.add_weighted_edges_from(edgeList)
    nlG0 = nx.normalized_laplacian_matrix(G0)
    # set iteration criteria for converge
    adjOld = nlG0
    # set celltype criteria for converge
    listResultOld = [1 for i in range(zOut.shape[0])]

    # Fill the zeros before EM iteration
    # TODO: better implementation later, now we don't filling zeros for now
    if args.zerofillFlag:
        for nz_index in range(len(scData.nz_i)):
            # tmp = scipy.sparse.lil_matrix.todense(scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]])
            # tmp = np.asarray(tmp).reshape(-1)[0]
            tmp = scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]]
            reconOut[scData.nz_i[nz_index], scData.nz_j[nz_index]] = tmp
        recon = reconOut

    # Define resolution
    # Default: auto, otherwise use user defined resolution
    if args.resolution == 'auto':
        if zOut.shape[0] < 2000:
            resolution = 0.8
        else:
            resolution = 0.5
    else:
        resolution = float(args.resolution)
    #print('Resolution: '+str(resolution))

    # Clustering not used in this file, so commenting out this section as the embeddings are exported in REGNN
    #print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
    #                )+"---EM process starts")

    for bigepoch in range(0, args.EM_iteration):
        #print('---'+str(datetime.timedelta(seconds=int(time.time() -
        #                                               start_time)))+'---Start %sth iteration.' % (bigepoch))

        # Now for both methods, we need do clustering, using clustering results to check converge
        # Clustering: Get clusters
        if args.clustering_method == 'Louvain':
            listResult, size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            #print('Louvain cluster: '+str(k))
        elif args.clustering_method == 'LouvainK':  # <-------- this one is being used in calculate ARI
            listResult, size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            #print('Louvain cluster: '+str(k))
            k = round(k*resolution) if round(k*resolution) >= 3 else 2 # changing int(k*resolution) to rounding
            #print('Usage Cluster: '+str(k))
            clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'LouvainB':
            listResult, size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            #print('Louvain cluster: '+str(k))
            k = int(k*resolution) if int(k*resolution) >= 3 else 2
            clustering = Birch(n_clusters=k).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'KMeans':
            clustering = KMeans(n_clusters=args.n_clusters,
                                random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'SpectralClustering':
            clustering = SpectralClustering(
                n_clusters=args.n_clusters, assign_labels="discretize", random_state=0).fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method == 'AffinityPropagation':
            clustering = AffinityPropagation().fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'AgglomerativeClustering':
            clustering = AgglomerativeClustering().fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method == 'AgglomerativeClusteringK':
            clustering = AgglomerativeClustering(
                n_clusters=args.n_clusters).fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method == 'Birch':
            clustering = Birch(n_clusters=args.n_clusters).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'BirchN':
            clustering = Birch(n_clusters=None).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'MeanShift':
            clustering = MeanShift().fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'OPTICS':
            clustering = OPTICS(min_samples=int(
                args.k/2), min_cluster_size=args.minMemberinCluster).fit(zOut)
            listResult = clustering.predict(zOut)
        else:
            print("Error: Clustering method not appropriate")
        #print('---'+str(datetime.timedelta(seconds=int(time.time() -
        #                                               start_time)))+"---Clustering Ends")

        # If clusters more than maxclusters, then have to stop
        if len(set(listResult)) > args.maxClusterNumber or len(set(listResult)) <= 1:
            print("Stopping: Number of clusters is " +
                  str(len(set(listResult))) + ".")
            # Exit
            # return None
            # Else: dealing with the number
            listResult = trimClustering(
                listResult, minMemberinCluster=args.minMemberinCluster, maxClusterNumber=args.maxClusterNumber)

        # Debug: Calculate silhouette
        # measure_clustering_results(zOut, listResult)
        #print('Total Cluster Number: '+str(len(set(listResult))))
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        # Shortcut for methods development for spatial data, may change the whole codes if iteration does not work
        # TODO
        if args.EM_iteration==1:  
            break                           # ------------------------- Breaks here ----------------------------------------

        # Graph regulizated EM AE with Cluster AE, do the additional AE
        if not args.quickmode:
            # Each cluster has a autoencoder, and organize them back in iteraization
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Start Cluster Autoencoder.')
            clusterIndexList = []
            for i in range(len(set(listResult))):
                clusterIndexList.append([])
            # print(len(clusterIndexList))
            for i in range(len(listResult)):
                # print('Debug:'+str(len(set(listResult)))+'\t'+str(len(listResult))+'\t'+str(i))
                # print(listResult[i])
                assignee = listResult[i]
                # Avoid bugs for maxClusterNumber
                if assignee == args.maxClusterNumber:
                    assignee = args.maxClusterNumber-1
                clusterIndexList[assignee].append(i)

            reconNew = np.zeros(
                (scData.features.shape[0], scData.features.shape[1]))

            # Convert to Tensor
            reconNew = torch.from_numpy(reconNew)
            if args.precisionModel == 'Double':
                reconNew = reconNew.type(torch.DoubleTensor)
            elif args.precisionModel == 'Float':
                reconNew = reconNew.type(torch.FloatTensor)
            reconNew = reconNew.to(device)

            model.load_state_dict(ptstatus)

            for clusterIndex in clusterIndexList:
                reconUsage = recon[clusterIndex]
                scDataInter = scDatasetInter(reconUsage)
                train_loader = DataLoader(
                    scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)
                for epoch in range(1, args.cluster_epochs + 1):
                    reconCluster, originalCluster, zCluster = train(
                        epoch, EMFlag=True)
                count = 0
                for i in clusterIndex:
                    reconNew[i] = reconCluster[count, :]
                    count += 1
                # empty cuda cache
                del originalCluster
                del zCluster
                torch.cuda.empty_cache()

            # Update
            recon = reconNew
            ptstatus = model.state_dict()

            # Debug mem consumption
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

        # Use new dataloader
        scDataInter = scDatasetInter(recon)
        train_loader = DataLoader(
            scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)

        for epoch in range(1, args.EM_epochs + 1):
            recon, original, z = train(epoch, EMFlag=True)

        zOut = z.detach().cpu().numpy()

        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'---Start Prune')
        if args.adjtype == 'unweighted':
            if args.MVstrategy == 'C2S':
                # Only need to calculate Once
                #adjTarget, edgeList = generateAdj(zOut, graphType='spatialGrid', para='euclidean:8:Grid', adjTag=(args.useGAEembedding or args.useBothembedding or (bigepoch == int(args.EM_iteration)-1)), spatialMatrix = spatialMatrix)
                adjSource, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k)+':'+args.pruneTag, adjTag=(args.useGAEembedding or args.useBothembedding or (bigepoch == int(args.EM_iteration)-1)), spatialMatrix = spatialMatrix)    
            if args.MVstrategy == 'SandC':
                # TODO not optimize for iteration now
                print("TODO")
            elif args.MVstrategy == 'NA':
                adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k)+':'+args.pruneTag, adjTag=(args.useGAEembedding or args.useBothembedding or (bigepoch == int(args.EM_iteration)-1)), spatialMatrix =spatialMatrix)
        
        elif args.adjtype == 'weighted':
            adj, edgeList = generateAdjWeighted(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(args.k), adjTag=(args.useGAEembedding or args.useBothembedding or (bigepoch == int(args.EM_iteration)-1)), spatialMatrix =spatialMatrix)
        
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Prune Finished')
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        # Whether use GAE embedding
        if args.useGAEembedding or args.useBothembedding:
            zDiscret = zOut > np.mean(zOut, axis=0)
            zDiscret = 1.0*zDiscret
            if args.useGAEembedding:
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))
                if args.MVstrategy == 'C2S':
                    zOut = GAEembeddingMultiView(zDiscret, adjSource, adjTarget, args)
                elif args.MVstrategy == 'NA' or args.MVstrategy == 'SandC' or args.MVstrategy == 'hybrid':
                    zOut = GAEembedding(zDiscret, adj,spatialMatrix,edges,args)
                print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                               start_time)))+"---GAE embedding finished")
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))
            elif args.useBothembedding:
                if args.MVstrategy == 'C2S':
                    zOut = GAEembeddingMultiView(zDiscret, adjSource, adjTarget, args)
                elif args.MVstrategy == 'NA' or args.MVstrategy =='SandC' or args.MVstrategy =='hybrid':
                    zOut = GAEembedding(zDiscret, adj, args)
                zOut = np.concatenate((zOut, zEmbedding), axis=1)

        # Original save step by step
        if args.saveinternal:
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Start save internal results')
            reconOut = recon.detach().cpu().numpy()

            # Output
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Prepare save')
            # # # print('Save results with reconstructed shape:'+str(reconOut.shape)+' Size of gene:'+str(len(genelist))+' Size of cell:'+str(len(celllist)))
            # recon_df = pd.DataFrame(np.transpose(
            #     reconOut), index=genelist, columns=celllist)
            # recon_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_recon_'+str(bigepoch)+'.csv')
            emblist = []
            for i in range(zOut.shape[1]):
                emblist.append('embedding'+str(i))
            embedding_df = pd.DataFrame(zOut, index=celllist, columns=emblist)
            embedding_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_embedding_'+str(bigepoch)+'.csv')
            # graph_df = pd.DataFrame(
            #     edgeList, columns=["NodeA", "NodeB", "Weights"])
            # graph_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_graph_'+str(bigepoch)+'.csv', index=False)
            results_df = pd.DataFrame(
                listResult, index=celllist, columns=["Celltype"])
            results_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_results_'+str(bigepoch)+'.txt')

            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Save internal completed')

        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Start test converge condition')

        # Iteration usage
        # If not only use 'celltype', we have to use graph change
        # The problem is it will consume huge memory for giant graphs
        if not args.converge_type == 'celltype':
            Gc = nx.Graph()
            Gc.add_weighted_edges_from(edgeList)
            adjGc = nx.adjacency_matrix(Gc)

            # Update new adj
            adjNew = args.alpha*nlG0 + \
                (1-args.alpha) * adjGc/np.sum(adjGc, axis=0)

            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---New adj ready')

            # debug
            graphChange = np.mean(abs(adjNew-adjOld))
            graphChangeThreshold = args.converge_graphratio * \
                np.mean(abs(nlG0))
            print('adjNew:{} adjOld:{} G0:{}'.format(adjNew, adjOld, nlG0))
            print('mean:{} threshold:{}'.format(
                graphChange, graphChangeThreshold))

            # Update
            adjOld = adjNew

        # Check similarity
        ari = adjusted_rand_score(listResultOld, listResult)

        # Debug Information of clustering results between iterations
        # print(listResultOld)
        # print(listResult)
        print('celltype similarity:'+str(ari))

        # graph criteria
        if args.converge_type == 'graph':
            if graphChange < graphChangeThreshold:
                print('Converge now!')
                break
        # celltype criteria
        elif args.converge_type == 'celltype':
            if ari > args.converge_celltyperatio:
                print('Converge now!')
                break
        # if both criteria are meets
        elif args.converge_type == 'both':
            if graphChange < graphChangeThreshold and ari > args.converge_celltyperatio:
                print('Converge now!')
                break
        # if either criteria are meets
        elif args.converge_type == 'either':
            if graphChange < graphChangeThreshold or ari > args.converge_celltyperatio:
                print('Converge now!')
                break

        # Update
        listResultOld = listResult
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                        )+"---"+str(bigepoch)+"th iteration in EM Finished")

    # TODO
    # Shortcut for methods development for spatial data, may change the whole codes if iteration does not work
    # No imputation for EM_iteration == 1
    if args.EM_iteration!=1:

        # Use new dataloader
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                        )+"---Starts Imputation")
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))
        scDataInter = scDatasetInter(reconOri)
        train_loader = DataLoader(
            scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)

        stateStart = torch.load(ptfileStart)
        model.load_state_dict(stateStart['state_dict'])
        optimizer.load_state_dict(stateStart['optimizer'])
        # model.load_state_dict(torch.load(ptfileStart))
        # if args.aePara == 'start':
        #     model.load_state_dict(torch.load(ptfileStart))
        # elif args.aePara == 'end':
        #     model.load_state_dict(torch.load(ptfileEnd))

        # generate graph regularizer from graph
        # adj = adj.tolist() # Used for read/load
        # adjdense = sp.csr_matrix.todense(adj)

        # Better option: use torch.sparse
        if args.sparseImputation == 'nonsparse':
            # generate adj from edgeList
            if args.MVstrategy == 'C2S':
                adj = adjSource
            adjdense = sp.csr_matrix.todense(adj)
            adjsample = torch.from_numpy(adjdense)
            if args.precisionModel == 'Float':
                adjsample = adjsample.float()
            elif args.precisionModel == 'Double':
                adjsample = adjsample.type(torch.DoubleTensor)
            adjsample = adjsample.to(device)
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

            # generate celltype regularizer from celltype
            celltypesample = generateCelltypeRegu(listResult)
            celltypesample = torch.from_numpy(celltypesample)
            if args.precisionModel == 'Float':
                celltypesample = celltypesample.float()
            elif args.precisionModel == 'Double':
                celltypesample = celltypesample.type(torch.DoubleTensor)
            celltypesample = celltypesample.to(device)
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

        for epoch in range(1, args.EM_epochs + 1):
            recon, original, z = train(
                epoch, EMFlag=True, taskType='imputation', sparseImputation=args.sparseImputation)

        reconOut = recon.detach().cpu().numpy()
        if not args.noPostprocessingTag:
            threshold_indices = reconOut < args.postThreshold
            reconOut[threshold_indices] = 0.0

    # Output final results
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---All iterations finished, start output embeddings.')
    # Output imputation Results
    # recon_df = pd.DataFrame(np.transpose(reconOut),
    #                         index=genelist, columns=celllist)
    # recon_df.to_csv(args.outputDir+args.datasetName+'_recon.csv')
    emblist = []
    for i in range(zOut.shape[1]):
        emblist.append('embedding'+str(i))
    embedding_df = pd.DataFrame(zOut, index=celllist, columns=emblist)
    # embedding_df.to_csv(args.outputDir+args.datasetName+'_embedding.csv')
    embedding_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_'+args.MVop+'_'+args.Sk+'_'+args.SpruneTag+'_embedding.csv')
    graph_df = pd.DataFrame(edgeList, columns=["NodeA", "NodeB", "Weights"])
    graph_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_'+args.MVop+'_'+args.Sk+'_'+args.SpruneTag+'_graph.csv', index=False)
    results_df = pd.DataFrame(listResult, index=celllist, columns=["Celltype"])
    # results_df.to_csv(args.outputDir+args.datasetName+'_results.txt')
    results_df.to_csv(args.outputDir+args.datasetName+'_'+str(args.k)+'_'+args.knn_distance+'_'+args.pruneTag+'_'+args.pe_type+'_'+str(args.PEtypeOp)+'_'+str(args.PEalpha)+'_'+args.MVop+'_'+args.Sk+'_'+args.SpruneTag+'_results.txt')

    # Adjusting for REGNN built with scGNN
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+"---REGNN process finished")
