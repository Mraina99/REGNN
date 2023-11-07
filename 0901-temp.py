import pickle
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
import numpy as np
import torch


with open("/home/chenghao/resept/RESEPT/label",'rb') as f:
    labels=pickle.load(f)

with open("/home/chenghao/resept/RESEPT/predict",'rb') as f:
    predicts=pickle.load(f)

print('label length: ',len(labels),len(predicts))

# print(adjusted_rand_score(np.array(labels, dtype=np.long),np.array(predicts,dtype=np.long)))
print('ARI: ',metrics.adjusted_rand_score(labels,predicts))



spatialMatrix=torch.load('/home/chenghao/resept/RESEPT/spatialMatrix.tensor')
print(spatialMatrix.shape)
print(spatialMatrix)
