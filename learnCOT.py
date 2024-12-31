# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops

import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random
import time

#from google.colab import drive
from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
#%matplotlib inline
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
from plotly import subplots

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc

import re

# import comet_ml
import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root=os.path

# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# I can wrap any intermediate activation in a HookPoint and get a convenient
# way to add PyTorch hooks
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x

# Define network architecture
# I defined my own transformer from scratch so I'd fully understand each component
# - I expect this wasn't necessary or particularly important, and a bunch of this
# replicates existing PyTorch functionality

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))

    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]


# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

class TripleAttentionSmarter(nn.Module):
    def __init__(self, model, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., hadamard=True):
        super().__init__()
        self.model = model
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.hadamard = hadamard

        # self.w = nn.Linear(dim, dim * 7, bias=qkv_bias)
        self.w = nn.Linear(dim, dim * 5, bias=qkv_bias)
        self.o = nn.Linear(2*dim,dim)
        self.attn_drop = nn.Dropout(0.3)


    def forward(self, x, test=False):
        # B: batch size, N: number of tokens, C: channels
        B, N, C = x.shape
        #print(x.shape)
        H = self.num_heads
        D = C // H

        # Calculate the A, B, C, D, E, F, V tensors
        w = self.w(x).reshape(B, N, 5, H, D).permute(2, 0, 3, 1, 4)
        x=x.reshape(B,H,N,D)#.permute(2,0,3,1,4)
        #a, b, c, v1, v2 = w[0], w[1], w[2], w[3], w[4]
        a, b, c, v1, v2 = w[0], w[1], w[2], w[3],w[4]

        # Calculate the attention matrices
        # AB shape: (B, H, N_I, N_J)
        # CD shape: (B, H, N_I, N_K)
        # EF shape: (B, H, N_J, N_K)

        #X = (torch.einsum('bhij,bhjk->bhik',a,b.transpose(-2,-1)) * self.scale)#.exp().unsqueeze(-1)
        #Y = (torch.einsum('bhij,bhjk->bhik',c,d.transpose(-2,-1)) * self.scale)#.exp().unsqueeze(-1)
        #Z = (torch.einsum('bhij,bhjk->bhik',e,f.transpose(-2,-1)) * self.scale)#.exp().unsqueeze(-1)
        
        X = (a @ b.transpose(-2, -1)) * self.scale
        Y = (b @ c.transpose(-2, -1)) * self.scale
        Z = (c @ a.transpose(-2, -1)) * self.scale

        #X=X-X.amax(axis=[2],keepdim=True)#sum()#[0]
        #Y=Y-Y.amax(axis=[2,3],keepdim=True)#[0]
        #Z=Z-Z.amax(axis=[3],keepdim=True)#[0]

        X=X.exp().unsqueeze(-1)
        Y=Y.exp().unsqueeze(-1)
        Z=Z.exp().unsqueeze(-1)
        
        X=self.attn_drop(X)
        Y=self.attn_drop(Y)
        Z=self.attn_drop(Z)
        
        # Unsqueeze v to shape (B, H, N_J, 1, D)
        Vj = v1.unsqueeze(-2)
        # Unsqueeze v to shape (B, H, 1, N_K, D)
        Vk = v2.unsqueeze(-3)
        
        # Expand Vj and Vk to shape (B, H, N_J, N_K, D)
        #V = Vj.expand(-1, -1, -1, N, -1)*Vk.expand(-1, -1, N, -1, -1)
        
        Vj=Vj.expand(-1, -1, -1, N, -1)
        Vk=Vk.expand(-1, -1, N, -1, -1)
        V=torch.cat((Vj,Vk),dim=-1)


        up=torch.einsum('bhijd,bhjkd->bhikd', torch.einsum('bhijd,bhjkd->bhikd', X, Y*V),Z)
        down=torch.einsum('bhijd,bhjkd->bhikd', torch.einsum('bhijd,bhjkd->bhikd', X, Y),Z)
       
        x=torch.einsum('bhiid->bhid', up)/torch.einsum('bhiid->bhid', down)
        #x=x.transpose(2,-1)
        return self.o(x.reshape(B, N, 2*C))

class TripleAttention(nn.Module):
    def __init__(self, model, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., hadamard=True):
        super().__init__()
        self.model = model
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.hadamard = hadamard

        # self.w = nn.Linear(dim, dim * 7, bias=qkv_bias)
        self.w = nn.Linear(dim, dim * 8, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if self.hadamard:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = nn.Linear(2 * dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, test=False):
        # B: batch size, N: number of tokens, C: channels
        B, N, C = x.shape
        H = self.num_heads
        D = C // H

        # Calculate the A, B, C, D, E, F, V tensors
        w = self.w(x).reshape(B, N, 8, H, D).permute(2, 0, 3, 1, 4)
        a, b, c, d, e, f, v1, v2 = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]

        # Calculate the attention matrices
        # AB shape: (B, H, N_I, N_J)
        # CD shape: (B, H, N_I, N_K)
        # EF shape: (B, H, N_J, N_K)

        AB = (a @ b.transpose(-2, -1)) * self.scale
        CD = (c @ d.transpose(-2, -1)) * self.scale
        EF = (e @ f.transpose(-2, -1)) * self.scale

        # Unsqueeze and expand AB to shape (B, H, N_I, N_J, 1)
        AB_expanded = AB.unsqueeze(-1)

        # Unsqueeze and expand CD to shape (B, H, N_I, 1, N_K)
        CD_expanded = CD.unsqueeze(-2)

        # Unsqueeze and expand EF to shape (B, H, 1, N_J, N_K)
        EF_expanded = EF.unsqueeze(-3)

        # Compute the final tensor by broadcasting and summing
        # Shape: (B, H, N_I, N_J, N_K)
        A_IJK = AB_expanded + CD_expanded + EF_expanded

        # Unsqueeze v to shape (B, H, N_J, 1, D)
        Vj = v1.unsqueeze(-2)
        # Unsqueeze v to shape (B, H, 1, N_K, D)
        Vk = v2.unsqueeze(-3)

        # Expand Vj and Vk to shape (B, H, N_J, N_K, D)
        Vj_expanded = Vj.expand(-1, -1, -1, N, -1)
        Vk_expanded = Vk.expand(-1, -1, N, -1, -1)

        # Concatenate along the last dimension (D) to get shape (B, H, N_J, N_K, 2D)
        V = Vj_expanded * Vk_expanded

        # Considering A_IJK and V are input tensors with shapes:
        # A_IJK: (B, H, N_I, N_J, N_K)
        # V: (B, H, N_J, N_K, 2D)

        A_IJK_expanded = A_IJK.unsqueeze(-1) # Shape: (B, H, N_I, N_J, N_K, 1)
        V_expanded = V.unsqueeze(-4) # Shape: (B, H, 1, N_J, N_K, 2D)

        # Compute the final tensor by broadcasting and summing
        # Shape: (B, H, N_I, N_J, N_K, 2D)
        e_AIJK = (A_IJK_expanded-A_IJK_expanded.max()).exp()
        x = (e_AIJK * V_expanded).sum((-2, -3)) / e_AIJK.sum((-2, -3)) # FIXME: Add numerically stable softmax
        x = x.transpose(1, 2).reshape(B, N, C)

        # TODO: include attention dropout
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.W_K2 = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q2 = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V2 = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O2 = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        #attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        #attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked/np.sqrt(self.d_head)), dim=-1))
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_pre/np.sqrt(self.d_head)), dim=-1))
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out#+out2

# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

# Transformer Block
class Iteration(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.ln1 = LayerNorm(d_model, model=self.model)
        #self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.attn = TripleAttentionSmarter(self.model,d_model, num_heads)
        self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        #y = self.hook_resid_mid(self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        y=self.attn(x)+x
        y = self.hook_resid_post(self.hook_mlp_out(self.mlp((y))))
        #y= torch.nn.linear()
        #print('')
        #print(y.size())
        #print(x.size())
        #x = torch.cat([x,torch.mean(y,dim=1).unsqueeze(1)],dim=1)
        x = torch.cat([x,y[:,-1,:].unsqueeze(1)],dim=1)
        return x

# Full transformer
class COT(nn.Module):
    def __init__(self, iterations, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, cot_it=3, use_cache=False,use_ln=False):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache
        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.iterations=iterations
        self.it = Iteration(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self])
        self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln
        #self.project=torch.nn.Linear(d_vocab,1)

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for i in range(self.iterations):
            x = self.it(x)
            x = self.ln(x)
            #x=torch.cat([x,self.project(y).unsqueeze(1)],dim=1)
        x = self.ln(x)
        x = self.unembed(x)
        return x

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

# Helper functions
def cuda_memory():
    print(torch.cuda.memory_allocated()/1e9)

def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    logprobs.size()
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss



def full_loss(model, data):
    global p
    # Take the final position only
    logits = model([i[0] for i in data])[:, -p]
    labels = torch.tensor([i[1] for i in data])#.to('cuda')
    return cross_entropy_high_precision(logits, labels)




lr=5e-3 #@param
#lr=1e-2
weight_decay =  2e-3 #@param
p=20


num_epochs = 1000 #@param
save_models = False #@param
save_every = 100 #@param
# Stop training when test loss is <stopping_thresh
stopping_thresh = -1 #@param
seed = 0 #@param
batch_style = 'full'
d_vocab = p+1
n_ctx = p+1

n_comp= 2
d_model = p*2
num_layers = 1

d_mlp = 32 # *n_comp
num_heads = 1
assert d_model % num_heads == 0
d_head = d_model//num_heads

act_type = 'ReLU' #@param ['ReLU', 'GeLU']

Train=True



def gen_train_test(num, nu_comp, seed=0, size=5000):
    data=[]
    for d in range(size):
        dat=np.random.choice(num,num)
        i=0
        for nu in range(nu_comp):
            #print(i)
            j=dat[i]
            i=0
            i+=j
        #print(i)
        dat=np.concatenate((dat,[20]))
        data.append([dat,i])
        #break
    return data

train = gen_train_test(p,n_comp, seed)
test = gen_train_test(p, n_comp, seed)

accu1=[]
accu2=[]
if Train:
    print('Start')
    model = COT(iterations=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_ln=False, use_cache=False)
    model.to('cuda')
    print('Model defined')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))#,amsgrad=True)
    #optimizer=optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    lossFunction=torch.nn.CrossEntropyLoss()
    #lossFunction=torch.nn.MSELoss()
    run_name = f"COTcomposition_{int(time.time())}"
    print(f'Run name {run_name}')
    if save_models:
        os.mkdir(run_name)
        save_dict = {'model':model.state_dict(), 'train_data':train, 'test_data':test}
        torch.save(save_dict, run_name+'/init.pth')
    for epoch in range(num_epochs):
        #random.shuffle(train)

        data=train#[:int(0.2*len(train))]
        logits = model([i[0] for i in data])[:, -1]
        #logits = F.softmax(logits,dim=-1).sum(dim=-1)
        # logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
        labels = torch.tensor([i[1] for i in data]).to('cuda')
        labels=labels.to(torch.int64).to('cuda')
        # prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
        loss=lossFunction(logits+1e-10,labels)
        #loss = -torch.mean(prediction_logprobs)
        #output = loss(logprobs,labels)
        if epoch%100 == 0: print(f"{epoch}_{np.log(loss.item())}")#_{train_acc.item():.4f}_{test_acc.item():.4f}")
        #loss.requires_grad = True
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if epoch%100 == 0:
            err=0
            for example in train:
                err+=int(torch.sum(abs(torch.tensor(example[1])-model([example[0]])[:,-1].argmax(axis=-1))))
            print('Accuracy test:')
            print(err/len(train))
            accu1.append(err/len(train))
            err=0
            for example in test:
                err+=int(torch.sum(abs(torch.tensor(example[1])-model([example[0]])[:,-1].argmax(axis=-1))))
            print('Accuracy test:')
            print(err/len(test))
            accu2.append(err/len(test))

   #}
   #torch.save(save_dict, run_name+'/'+f"final.pth")
    print(f"Done")
    # lines([train_losses, test_losses], labels=['train', 'test'], log_y=True)

    # save_models = False

if not(Train):
    loaded=torch.load("Att4Pairs_1719607430/final.pth")
    model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head, num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)
    model.load_state_dict(loaded['model'])

plt.plot(np.arange(1000,step=100),accu1)
plt.plot(np.arange(1000,step=100),accu2)
plt.show()

#test = gen_train_test(p, seed)

err=0
for example in test:
    err+=int(abs(example[1]-model([example[0]])[:,-1].argmax()))
print(1.0*err/len(test))




