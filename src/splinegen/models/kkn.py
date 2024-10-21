import torch
from torch import nn
import math
from .decoder_layer import TransformerDecoderForKPN

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

class KPN_Discrete(nn.Module):

    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout, output_dim):

        super(KPN_Discrete, self).__init__()
        self.embedding_tgt = nn.Linear(output_dim, d_model) # 1 + 2 (number of Tokens)


        self.pos_encoder_output = PositionalEncoding(d_model, dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # Projection
        self.fully_connected = nn.Linear(d_model,output_dim)

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
        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder_output(tgt)

        max_knots_len = tgt.size(1)
        tgt_mask_lower_tridiagonal = nn.Transformer.generate_square_subsequent_mask(max_knots_len,device=tgt.device)
        # Pass through the transformer
        # transformer_output = self.transformer(src, tgt, tgt_mask= ,src_key_padding_mask=src_key_padding_mask)
        # transformer_output = self.decoder(tgt,src, tgt_mask=tgt_mask_lower_tridiagonal, tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
        transformer_output = self.decoder(tgt,src, tgt_mask=tgt_mask_lower_tridiagonal, tgt_key_padding_mask=tgt_key_padding_mask)

        fc_output = self.fully_connected(transformer_output)
        knots_tmp = fc_output
        knots = knots_tmp

        return knots

class KPN(nn.Module):

    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout,output_dim=3 ):

        super(KPN, self).__init__()
        # self.max_seq_len = max_seq_len
        # self.max_knots_len = max_knots_len

        # Embedding layer,
        # Mapping from target space to d_model
        self.nhead=nhead
        self.embedding_tgt = nn.Linear(output_dim, d_model) # 1 + 2 (number of Tokens)


        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer

        # self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,

        #                                   batch_first=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        # Linear layer to project to desired shape
        # self.fc = nn.Linear(max_seq_len * d_model, max_knots_len - 2 * (p + 1))
        # self.fc = nn.Linear(max_seq_len * d_model, max_knots_len)

        # Projection
        self.fully_connected = nn.Linear(d_model, output_dim)

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

        # tgt = torch.unsqueeze(tgt, -1)
        # tgt (batch_size*max_knots_len*3) -> (batch_size*max_knots_len*d_model)
        tgt = self.embedding_tgt(tgt)
        # tgt = self.pos_encoder(tgt)

        max_knots_len = tgt.size(1)
        tgt_mask_lower_tridiagonal = nn.Transformer.generate_square_subsequent_mask(max_knots_len, device='cuda')

        memory_mask=src_key_padding_mask.unsqueeze(1).expand(-1,max_knots_len,-1).repeat((self.nhead, 1, 1))
        
        # Pass through the transformer
        # transformer_output = self.transformer(src, tgt, tgt_mask= ,src_key_padding_mask=src_key_padding_mask)
        transformer_output = self.decoder( 
            tgt,src, tgt_mask=tgt_mask_lower_tridiagonal,  
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_mask=memory_mask)

        # Reshape and pass through the fully connected layer.
        # (batch_size*max_knots_len*d_model) -> (batch_size*max_knots_len*1) -> (batch_size*max_knots_len)
        # batch_size = transformer_output.size(0)
        # (batch_size*max_knots_len*d_model) -> (batch_size*max_knots_len*3)
        fc_output = self.fully_connected(transformer_output)
        # fc_output = fc_output.view(batch_size, -1)

        # use tgt_key_padding_mask, set padding places to neg inf
        neg_inf = torch.tensor(float('-inf'))
        neg_inf_mask = torch.zeros_like(fc_output)
        #########
        tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(-1)
        tgt_key_padding_mask = tgt_key_padding_mask.expand(-1, -1, 3)
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
        # my_function = torch.nn.ReLU()
        # my_function = torch.nn.Tanh()

        knots = my_function(knots_tmp)
        #################

        # Accumulate Increments
        # interior_knots = torch.cumsum(increments, dim=-1)
        # knots = torch.cumsum(increments, dim=-1)
        ############
        tgt_mask_position = tgt_mask_position.unsqueeze(-1)
        tgt_mask_position = tgt_mask_position.expand(-1, -1, 3)
        knots = knots * tgt_mask_position

        # Add p+1 zeros at the beginning and p+1 ones at the end
        # zeros = torch.zeros(src.size(0), self.p + 1, device=src.device)
        # ones = torch.ones(src.size(0), self.p + 1, device=src.device)
        # knots = torch.cat([zeros, interior_knots, ones], dim=-1)

        return knots,transformer_output

class KPN2(nn.Module):

    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout,output_dim=3 ):

        super(KPN2, self).__init__()
        # self.max_seq_len = max_seq_len
        # self.max_knots_len = max_knots_len

        # Embedding layer,
        # Mapping from target space to d_model

        self.embedding_tgt = nn.Linear(output_dim, d_model) # 1 + 2 (number of Tokens)


        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer

        # self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,

        #                                   batch_first=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderForKPN(decoder_layer, num_decoder_layers, decoder_norm)
        self.nhead=nhead
        # Linear layer to project to desired shape
        # self.fc = nn.Linear(max_seq_len * d_model, max_knots_len - 2 * (p + 1))
        # self.fc = nn.Linear(max_seq_len * d_model, max_knots_len)

        # Projection
        self.fully_connected = nn.Linear(d_model, output_dim)

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

        # tgt = torch.unsqueeze(tgt, -1)
        # tgt (batch_size*max_knots_len*3) -> (batch_size*max_knots_len*d_model)
        tgt = self.embedding_tgt(tgt)
        # tgt = self.pos_encoder(tgt)

        max_knots_len = tgt.size(1)
        tgt_mask_lower_tridiagonal = nn.Transformer.generate_square_subsequent_mask(max_knots_len, device='cuda')

        memory_mask=src_key_padding_mask.unsqueeze(1).expand(-1,max_knots_len,-1).repeat((self.nhead, 1, 1))
        
        # Pass through the transformer
        # transformer_output = self.transformer(src, tgt, tgt_mask= ,src_key_padding_mask=src_key_padding_mask)
        transformer_output ,last_output,kv= self.decoder(
            tgt,src, tgt_mask=tgt_mask_lower_tridiagonal,  tgt_key_padding_mask=tgt_key_padding_mask,memory_mask=memory_mask)

        # Reshape and pass through the fully connected layer.
        # (batch_size*max_knots_len*d_model) -> (batch_size*max_knots_len*1) -> (batch_size*max_knots_len)
        # batch_size = transformer_output.size(0)
        # (batch_size*max_knots_len*d_model) -> (batch_size*max_knots_len*3)
        fc_output = self.fully_connected(transformer_output)
        # fc_output = fc_output.view(batch_size, -1)

        # use tgt_key_padding_mask, set padding places to neg inf
        neg_inf = torch.tensor(float('-inf'))
        neg_inf_mask = torch.zeros_like(fc_output)
        #########
        tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(-1)
        tgt_key_padding_mask = tgt_key_padding_mask.expand(-1, -1, 3)
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
        # my_function = torch.nn.ReLU()
        # my_function = torch.nn.Tanh()

        knots = my_function(knots_tmp)
        #################

        # Accumulate Increments
        # interior_knots = torch.cumsum(increments, dim=-1)
        # knots = torch.cumsum(increments, dim=-1)
        ############
        tgt_mask_position = tgt_mask_position.unsqueeze(-1)
        tgt_mask_position = tgt_mask_position.expand(-1, -1, 3)
        knots = knots * tgt_mask_position

        # Add p+1 zeros at the beginning and p+1 ones at the end
        # zeros = torch.zeros(src.size(0), self.p + 1, device=src.device)
        # ones = torch.ones(src.size(0), self.p + 1, device=src.device)
        # knots = torch.cat([zeros, interior_knots, ones], dim=-1)

        return knots,last_output,kv