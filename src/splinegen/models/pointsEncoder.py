import torch
import torch.nn.functional as F

from torch import nn
import math

class PointsEncoder(nn.Module):
    '''
        using a transformer encoder
    '''
    def __init__(self,input_dim,hidden_dim,num_layers,num_head,dim_feedforward,dropout,frequence_band=10,**kwargs) -> None:
        super().__init__()
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim,dropout=dropout,dim_feedforward=dim_feedforward,nhead=num_head,batch_first=True)
        layer_norm=nn.LayerNorm(hidden_dim)
        self.encoder=nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=num_layers,norm=layer_norm,enable_nested_tensor=False)
        self.position_encoder=PositionalEncoding(hidden_dim)

        self.linear_projection=nn.Linear(2*frequence_band*input_dim,hidden_dim)
        self.dropout=nn.Dropout(dropout)
        self.L=frequence_band

    def forward(self,x,mask):
        '''
            x: [batch,seq_len,hidden_dim]
        '''
        x=coordinates_positional_encoding(x,self.L)
        # torch._assert(x.shape[1]==mask.shape[1],f"the length of x and mask should be the same, but got {x.shape[1]} and {mask.shape[1]}")
        x=self.linear_projection(x)
        x=self.dropout(x)
        # x=self.position_encoder(x)
        x_ = self.encoder(x,src_key_padding_mask=torch.logical_not(mask))
        # torch._assert(x_.shape[1]==mask.shape[1],f"the length of x and mask should be the same, but got {x_.shape[1]} and {mask.shape[1]}")

        return x_

def coordinates_positional_encoding(x, L=10):
    """
    Applies positional encoding to the input tensor `x`.
    
    Args:
        x: A tensor of shape [..., D] where `...` represents any number of
           preceding dimensions and `D` is the dimensionality of the input
           to be positionally encoded.
        L: The number of frequency bands used in the positional encoding.
    
    Returns:
        A tensor of shape [..., D * 2 * L] containing the positionally
        encoded representation.
    """

    # Create a list of frequencies
    frequencies = 2.0 ** torch.arange(0, L, dtype=x.dtype, device=x.device)

    # View the frequencies to make them broadcastable with x
    frequencies = frequencies.view(*([1] * len(x.shape)), L)

    # Apply the positional encoding by concatenating sines and cosines
    x_expanded = x[..., None] * frequencies  # Shape: [..., L, D]
    x_encoded = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
    
    # Reshape the encoded tensor to have the correct size
    # shape = list(x.shape[:-1]) + [-1]
    # x_encoded = x_encoded.reshape(shape)
    x_encoded = x_encoded.flatten(start_dim=-2)

    return x_encoded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough `pe` matrix that can be sliced according to maximum sequence lengths encountered
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        # Take the positional encodings up to the sequence length of the input
        x = x + self.pe[:, :x.size(1)]
        return x

class SimplePointsEncoder(nn.Module):
    def __init__(self, en_input_dim,en_d_model,en_dropout=0.01) -> None:
        super().__init__()
        self.points_encoder=nn.Sequential(nn.Linear(en_input_dim,en_d_model),nn.Dropout(en_dropout))
        # self.points_encoder=nn.Linear(en_input_dim,en_d_model)

    def forward(self,src,src_mask):
        return self.points_encoder(src)