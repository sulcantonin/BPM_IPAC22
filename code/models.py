import torch 
import torch.nn as nn
import torch.nn.functional as F
import gzip
import pickle
import numpy as np
import math
from tqdm import tqdm

# NOTE!
# In the original paper, $c$ is kept fixed and all layers are bias-free becauase there is a trivial solution, (c=0 or c=b)
# The standard TransformerEncoderLayer uses biases by default and it is not possible to disable them
# Use TransformerEncder.py instead. It is much simpler, but bias free model. 

class TransformerModelTorch(nn.Module):
    def __init__(self, n_in, latent_dim, nhead = 1, dropout_p = 0.0):
        super(TransformerModelTorch, self).__init__()
       
        self.layer1 = nn.TransformerEncoderLayer(d_model = n_in, nhead = nhead, dim_feedforward = n_in, dropout = dropout_p)
        self.layer2 = nn.TransformerEncoderLayer(d_model = n_in, nhead = nhead, dim_feedforward = n_in, dropout = dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.output = nn.Linear(n_in,latent_dim)
        self.c = nn.Parameter(torch.randn(latent_dim), requires_grad = True) # 2022/08/20 fix -> this shouldn't be updatable parameter, because otherwise, the biases of transformer mutually converge to c = b 
        # self.c = torch.randn(latent_dim, requires_grad = False) # 2022/08/20 fix -> this shouldn't be updatable parameter, because otherwise, the biases of transformer mutually converge to c = b 
        
    
    def hidden_fro_norm(self):
        norm = []
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                norm.append(torch.linalg.matrix_norm(param.view((-1,1))))
        return torch.stack(norm).mean()
    
    def phi(self, x):
        if x.dim() == 2: # !TODO dirty hack
            x = x.unsqueeze(1)
        assert x.dim() == 3
        # x = self.pe(x)
        return self.output(self.dropout2(self.layer2(self.dropout1(self.layer1(x)))))
    
    def forward(self, x):
        phi = self.phi(x)
        return torch.sqrt(torch.sum((phi - self.c) ** 2,-1))

    
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
    

def get_X_from_file(file, full_adcs):
    '''
    helper function for loading data from files sase files (only first bunches)
    '''
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
