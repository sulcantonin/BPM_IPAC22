import torch 
import torch.nn as nn
import torch.nn.functional as F
import gzip
import pickle
import numpy as np
import math
from tqdm import tqdm

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros((max_len, d_model),requires_grad = False)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        '''
        if x.min() < 0.0:
            x -= x.min() # (-inf,inf) -> (0,inf)
        x /= x.max() # (0,inf) -> (0,1)
        x = (x - 0.5) * 2 # (0,1) -> (-1,1)
        '''
        
        '''
        x = x[:,0,:]
        x -= x.min(axis=-1,keepdims = True)[0]
        x /= x.max(axis=-1,keepdims = True)[0]
        x = (x-0.5) * 2
        x = x.unsqueeze(1)
        x = x + self.pe[:x.size(0)]
        '''
        pe = self.layer_norm(self.pe[:x.size(0),:]) * x.std()
        x = x + pe
        return self.dropout(x)

    
# 
# Attention 
# 
class AttentionModule(nn.Module):
    def __init__(self, D, D_qk, D_v, dropout_p = 0.0):
        super(AttentionModule, self).__init__()

        self.phi_q = nn.Linear(D,D_qk, bias = False) # the attention is just matrix multiplication, no bias
        self.phi_k = nn.Linear(D,D_qk, bias = False) # the same here
        self.phi_v = nn.Linear(D,D_v, bias = False)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dq_sqrt_inv = np.sqrt(float(D_qk))
        
    def self_attention(self, x):
        return F.softmax(self.dq_sqrt_inv * self.phi_q(x) @ self.phi_k(x).T, dim = -1)
    
    def forward(self,x):
        return self.self_attention(x) @ (self.phi_v(x))

class TransformerLayer(nn.Module):
    def __init__(self, n_in, n_out, D_qk, D_v, bias = False):
        super(TransformerLayer, self).__init__()
        self.n_in = n_in
        self.self_attn = AttentionModule(n_in,D_qk, D_v)
        self.layer_norm1 = nn.LayerNorm(n_in, elementwise_affine = False)
        self.linear = nn.Linear(n_in, n_out, bias = bias)
        self.layer_norm2 = nn.LayerNorm(n_in, elementwise_affine = False)

    def forward(self, x):
        # based on the eq. 8 in https://www.borealisai.com/research-blogs/tutorial-14-transformers-i-introduction/
        sa = self.self_attn(x)
        '''
        x = sa + x
        x = self.layer_norm1(x)
        x = sa + self.linear(x)
        x = self.layer_norm2(x)
        return x
        '''
        return self.layer_norm2(sa + self.linear(self.layer_norm1(x + sa)))

class TransformerModel(nn.Module):
    def __init__(self, n_in, D_qk, latent_dim, dropout_p = 0.0, bias = False, use_pe = False, normalize_input = False):
        super().__init__()
        self.n_in = n_in
        self.layer1 = TransformerLayer(n_in, n_in, D_qk, n_in, bias = bias)
        # self.layer2 = TransformerLayer(n_in, n_in, D_qk, n_in, bias = bias)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.output = nn.Linear(n_in, latent_dim, bias = False)
        self.c = nn.Parameter(torch.randn(latent_dim),requires_grad = False)
        self.normalize_input = normalize_input
        # self.use_pe = nn.Parameter(torch.tensor(use_pe), requires_grad = False)
        self.use_pe = use_pe
        if use_pe:
            self.pe = PositionalEncoding(n_in, dropout_p, 700)
        else:
            self.pe = None
        
        
    def phi(self, x):
        
        if self.normalize_input:
            x = F.layer_norm(x,(self.n_in,))
        if self.use_pe:
            x = self.pe(x)
        return self.output(self.dropout1(self.layer1(x)))
        # return self.output(self.dropout2(self.layer2(self.dropout1(self.layer1(x)))))

    def forward(self, x):
        y = self.phi(x)
        return torch.linalg.vector_norm(y - self.c,dim = -1)


    
def show_input_gradients_for_output(X, model, device):
    '''
    Inspired by https://discuss.pytorch.org/t/newbie-getting-the-gradient-with-respect-to-the-input/12709
    
    val = show_input_gradients_for_output(X[::100,:],model,device)
    plt.imshow(val)
    plt.show()    
    
    '''
    rev_scores = []
    for i in tqdm(range(1,X.shape[0])):
        x = torch.autograd.Variable(X[:i,:],requires_grad=True).to(device)
        rev_score = torch.autograd.grad(model(x)[-1],x)
        rev_scores.append(rev_score[0][-1:,...])
    return torch.cat(rev_scores).detach().cpu().numpy()
    
# helper function for loading data from files sase files (only first bunches)
def get_X_from_file(file, full_adcs):
    with gzip.open(file,'rb') as f:
        data = pickle.load(f)
        x = data['first_bunch_x'][full_adcs]
        y = data['first_bunch_y'][full_adcs]

        X = np.stack( (x.to_numpy(),y.to_numpy()),0).astype(np.float16)
        X = X - X.mean(1,keepdims = True)
        
        # removing bunches with nans
        bunches_without_nans = np.isfinite(X.sum((0,2)))
        X = X[:,bunches_without_nans,:]
        
        # other operations
        dim, N, vals = X.shape
        X = X.transpose((1,0,2))
        X = X.reshape((N,1,dim * vals))
        X = torch.tensor(X)
    return X

def exclude_time_ranges(data_folder, data_files, excluded_time_ranges):
    '''
    Removes list of predefined time ranges (e.g. empty, downtime whatever) time ranges
    
    Example:
    
    > excluded_time_ranges = [
    >    {'start' : datetime(2022,3,23,18,40,0), 'stop' : datetime(2022,3,28,7,10,0), 'step' : timedelta(minutes = 1)}]
    > data_folder = '/blah/blah/'
    > data_files = sorted(glob(data_folder + '*.pickle'))
    > data_files = exclude_time_ranges(data_folder, data_files, excluded_time_ranges)
    '''
    
    compile_path = lambda folder, time : folder + time.strftime('%Y%m%d_%H%M%S') + '.pickle'
    
    for etr in excluded_time_ranges:
        excluded_file_set = set()
        start = etr['start']
        while start <= etr['stop']:
            excluded_file_set.add(compile_path(data_folder, start))
            start += etr['step']
        data_files = list(filter(lambda file : not file in excluded_file_set,data_files))
    return data_files
