import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, D, D_qk, D_v, dropout_p = 0.0, bias = False):
        super(AttentionModule, self).__init__()

        self.phi_q = nn.Linear(D,D_qk, bias = bias)
        self.phi_k = nn.Linear(D,D_qk, bias = bias)
        self.phi_v = nn.Linear(D,D_v, bias = bias)
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
        self.attention = AttentionModule(n_in,D_qk, D_v, bias = bias)
        self.layer_norm1 = nn.LayerNorm(n_in)
        self.linear = nn.Linear(n_in, n_out, bias = bias)
        self.layer_norm2 = nn.LayerNorm(n_in)
    def forward(self, X):
        return self.layer_norm2(X + self.linear(self.layer_norm1(X + self.attention(X))))
    
class TransformerModel(nn.Module):
    def __init__(self, n_in, D_qk, latent_dim, dropout_p = 0.0, bias = False):
        super(TransformerModel, self).__init__()
        self.layer1 = TransformerLayer(n_in, n_in, D_qk, n_in, bias = bias)
        self.layer2 = TransformerLayer(n_in, n_in, D_qk, n_in, bias = bias)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.output = nn.Linear(n_in, latent_dim)
        self.c = torch.randn(latent_dim)
    
    def phi(self, x):
        return self.output(self.dropout2(self.layer2(self.dropout1(self.layer1(x)))))
    
    def forward(self, x):
        y = self.phi(x)
        return torch.sqrt(torch.sum((y - self.c) ** 2,-1))
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.layer_norm = nn.LayerNorm(d_model)

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
        
        # x = torch.cat((x,self.pe[:x.size(0)]),-1)
        x = x + self.layer_norm(self.pe[:x.size(0)])
        return self.dropout(x)
