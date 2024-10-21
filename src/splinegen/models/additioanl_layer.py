from torch import nn
import math
import torch

class AdditionalLayer(nn.Module):

    def __init__(self,degree):
        super().__init__()
        self.degree=degree
        self.finetune=TunableLayer(input_dim=4,d_model=512,nhead=4,num_decoder_layers=3,num_encoder_layers=3,dim_feedforward=2048,dropout=0.01,p=3,max_knots_len=30,max_seq_len=30)
        self.finetune2=TunableLayer(input_dim=1,tgt_input_dim=4,d_model=512,nhead=4,num_decoder_layers=3,num_encoder_layers=3,dim_feedforward=2048,dropout=0.01,p=3,max_knots_len=30,max_seq_len=30)
        # self.finetune2=PPNFineTune(input_dim=1,head=False,tgt_input_dim=3,d_model=256,nhead=4,num_decoder_layers=4,num_encoder_layers=4,dim_feedforward=1024,dropout=0.01,p=3,max_knots_len=30,max_seq_len=30)
        
    def forward(self,points,params,points_mask,knots,knots_mask):
        '''
            x is data dict
        '''
        
        src_input=torch.cat([params.unsqueeze(-1),points],dim=-1)
        # new_params=x['params']+1e-1*self.finetune2(x['knots'].unsqueeze(-1),x['knots_mask'],src_input,x['points_mask'])
        # new_params=torch.clip(new_params,0,1)
        new_params=self.finetune2(knots.unsqueeze(-1),knots_mask,src_input,points_mask)

        new_knots=self.finetune(src_input,points_mask,knots.unsqueeze(-1),knots_mask)
        new_knots=1e-1*new_knots.squeeze(-1)+knots
        new_knots=torch.clip(new_knots,0.01,0.99)
        # new_knots=torch.nn.functional.sigmoid(new_knots)

        p=self.degree
        max_knots_len=new_knots.shape[-1]
        batch_size=new_knots.shape[0]
        index=torch.arange(max_knots_len, dtype=torch.long,device=points.device)+p+2
        index=index.clip(max=max_knots_len-1)
        mask=torch.gather(knots_mask, 1, index.unsqueeze(0).expand(batch_size,-1))
        mask[:,-(p+1):]=False
            
        output=torch.zeros_like(new_knots)
        output[:,p+1:]=new_knots[:,p+1:]

        output=output.masked_fill_(~mask,1)
        # train_output=torch.masked_fill(output, ~mask, 1)
        output_sorted=torch.sort(output,dim=-1)[0]
        knots=output_sorted
        params=new_params
        # loss,ctrl=nurbs_eval.getCtrlPts(self.degree,[x['params'],x['points'],x['points_mask'],x['knots'],x['knot_length']-1])
        
        return params,knots

class TunableLayer(nn.Module):

    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, p, max_knots_len, max_seq_len,tgt_input_dim=1,tgt_out_dim=1,head=True):

        super(TunableLayer, self).__init__()
        self.input_dim = input_dim  # input_dim is 2 or 3
        self.p = p
        # self.max_seq_len = max_seq_len
        # self.max_knots_len = max_knots_len

        # Embedding layer,
        self.embedding = nn.Linear(input_dim, d_model)
        # Mapping from target space to d_model

        self.embedding_tgt = nn.Linear(tgt_input_dim, d_model) # 1 + 2 (number of Tokens)


        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,

                                          batch_first=True)

        # Linear layer to project to desired shape
        # self.fc = nn.Linear(max_seq_len * d_model, max_knots_len - 2 * (p + 1))
        # self.fc = nn.Linear(max_seq_len * d_model, max_knots_len)

        # Projection
        self.fully_connected = nn.Linear(d_model, tgt_out_dim)

        self.head=head

    def forward(self, src, src_mask_position, tgt, tgt_mask_position):
        '''
        input: 
            src: cat points and parameters, (Batch_Size, max_seq_len, dim of points and parameters)
            src_mask_position: src position mask, meaningful position are Ture, (Batch_Size, max_seq_len)
            tgt: knots shifted right, (Batch_Size, max_knots_len, 3), the third channel have three dimensions, which indicate knots, SOS, EOS. 
            tgt_mask_position : tgt position mask, should be shifted right, meaningful position are Ture, (Batch_Size, max_knots_len)
        output:
            train_output: knots, SOS, EOS. shifted left (Batch_Size, max_knots_len, 3)
        
        *SOS means Start of Sentence
        *EOS means End of Sentence
        '''
        
        src_key_padding_mask = ~src_mask_position
        tgt_key_padding_mask = ~tgt_mask_position

        # src (batch_size*max_seq_len*(3+2)) -> (batch_size*max_seq_len*d_model)
        src = self.embedding(src)  # Apply embedding
        # src = self.pos_encoder(src)
        # tgt (batch_size*max_knots_len) -> (batch_size*max_knots_len,1) -> (batch_size*max_knots_len*d_model)
        # tgt = torch.unsqueeze(tgt, -1)
        # tgt (batch_size*max_knots_len*3) -> (batch_size*max_knots_len*d_model)
        tgt = self.embedding_tgt(tgt)
        # tgt = self.pos_encoder(tgt)

        max_knots_len = tgt.size(1)

        # Pass through the transformer
        # transformer_output = self.transformer(src, tgt, tgt_mask= ,src_key_padding_mask=src_key_padding_mask)
        transformer_output = self.transformer(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # Reshape and pass through the fully connected layer.
        # (batch_size*max_knots_len*d_model) -> (batch_size*max_knots_len*1) -> (batch_size*max_knots_len)
        batch_size = transformer_output.size(0)
        # (batch_size*max_knots_len*d_model) -> (batch_size*max_knots_len*3)
        fc_output = self.fully_connected(transformer_output)
        # fc_output = fc_output.view(batch_size, -1)

        # use tgt_key_padding_mask, set padding places to neg inf
        neg_inf = torch.tensor(float('-inf'))
        neg_inf_mask = torch.zeros_like(fc_output)
        #########
        tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(-1)
        tgt_key_padding_mask = tgt_key_padding_mask.expand(-1, -1, 1)
        ########
        neg_inf_mask.masked_fill_(tgt_key_padding_mask, neg_inf)
        # increments_tmp = fc_output + neg_inf_mask

        # softmax ensure the increment are in [0,1], and sum up to 1
        # increments = softmax(increments_tmp, dim=-1)

        
        # my_function = torch.nn.Sigmoid()
        # increments = my_function(increments_tmp)
        
        #############
        knots_tmp = fc_output + neg_inf_mask
        my_function = torch.nn.Sigmoid()
        # my_function=torch.nn.ReLU()
        # my_function = torch.nn.Tanh()

        if self.head:
            knots = my_function(knots_tmp)
        else:
            knots=knots_tmp
        #################

        # Accumulate Increments
        # interior_knots = torch.cumsum(increments, dim=-1)
        # knots = torch.cumsum(increments, dim=-1)
        ############
        tgt_mask_position = tgt_mask_position.unsqueeze(-1)
        tgt_mask_position = tgt_mask_position.expand(-1, -1, 1)
        knots = knots * tgt_mask_position

        # Add p+1 zeros at the beginning and p+1 ones at the end
        # zeros = torch.zeros(src.size(0), self.p + 1, device=src.device)
        # ones = torch.ones(src.size(0), self.p + 1, device=src.device)
        # knots = torch.cat([zeros, interior_knots, ones], dim=-1)

        return knots.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Adjust for batch-first input
        return self.dropout(x)