# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 09:48:20 2024

@author: h
"""

import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset

def triangle_free(adj):
    sqn=np.shape(adj)[0]
    for a in range(sqn):
        for b in range(sqn):
            if adj[a,b]==1 and not(a==b):
                for c in range(sqn):
                    if not(a==c) and not(b==c):
                        if adj[b,c]==1 and adj[a,c]==1:
                            return [1,0]#False
    return [0,1]#True

def flatten(adj):
    #print(np.shape(adj))
    #print(len(adj[np.tril_indices(np.shape(adj)[0],-1)]))
    return adj.reshape((np.shape(adj)[0]*np.shape(adj)[1]))
    #return adj[np.tril_indices(np.shape(adj)[0],-1)]


def gen_validation(num,p, mu=2):
    #data=[]
    sqn=int(np.sqrt(num))
    #inds=np.arange(sqn)
    x=torch.zeros((mu*500,sqn**2))
    y=torch.zeros((mu*500,2))
    for sample in range(mu*500):
        adj=np.zeros((sqn,sqn))
        for a in range(sqn):
            for b in range(sqn):
                adj[a,b]=random.choices([0,1],[1-p,p],k=1)[0]
        adj=np.tril(adj.T,-1) + np.triu(adj, 1)
        adj[np.diag_indices_from(adj)]=0
        x[sample,:]=torch.tensor(flatten(adj))
        #break
        y[sample,:]=torch.tensor(triangle_free(adj))
    return TensorDataset(x,y)




import random
nodes=range(9,9+20)
valus=[0.25,0.225,0.2,0.175,0.15,0.125,0.1,0.09,0.075,0.05,0.025,0.01,0.009,0.008,0.007,0.006,0.005]
#l=[]
pairs=[]
for node in nodes:
    l=[]
    for val in valus:
        
        test=gen_validation(node*node,val)
        c=0
        for k in range(len(test)):
            if torch.argmax(test[k][1])==1:
                c+=1
        l.append(abs(0.5-c/len(test)))
    #print(l)
    i=np.argmin(np.array(l))
    print((node,i))
    pairs.append((node,valus[i]))
    
    
    