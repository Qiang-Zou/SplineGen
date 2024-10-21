from torch import nn
from torchvision.ops.misc import MLP

class paramDecoderMLP(nn.Module):
    def __init__(self, c_input:int,hidden_dim:int,dropout=0.1,output_dim=1) -> None:
        super().__init__()
        self.c_input = c_input
        self.hidden_dim = hidden_dim
        self.output_dim=output_dim
        self.mlp=MLP(c_input,hidden_channels=[hidden_dim,hidden_dim,output_dim],norm_layer=nn.LayerNorm,dropout=dropout,activation_layer=nn.ReLU)

    def forward(self,x,mask):
        return self.mlp(x)

class paramDecoderTransformer(nn.Module):
    def __init__(self, c_input:int,hidden_dim:int,dropout=0.1) -> None:
        super().__init__()
        self.c_input = c_input
        self.hidden_dim = hidden_dim
        self.mlp=nn.TransformerEncoder(nn.TransformerEncoderLayer(c_input, 4, hidden_dim, dropout), 1)
    def forward(self,x,mask):
        return self.mlp(x)