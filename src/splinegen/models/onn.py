import torch
from torch import nn
from typing import Tuple, Union, Optional
from .decoder_layer import TransformerDecoderLayerWithKnots as TransformerDecoderLayerWithKnots,TransformerDecoderWithKnots as TransformerDecoderWithKnots

# from .param_decoder import paramDecoder

TOKENS = {
  '<eos>': 0
}

def info_value_of_dtype(dtype: torch.dtype):
  if dtype == torch.bool:
    raise TypeError("Does not support torch.bool")
  elif dtype.is_floating_point:
    return torch.finfo(dtype)
  else:
    return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
  return info_value_of_dtype(dtype).min

def masked_log_softmax(
  x: torch.Tensor,
  mask: torch.Tensor,
  dim: int = -1,
  eps: float = 1e-45
) -> torch.Tensor:

  x = x + (mask.float() + eps).log()
  return torch.nn.functional.log_softmax(x, dim=dim)
 
def masked_max(
  x: torch.Tensor,
	mask: torch.Tensor,
	dim: int,
	keepdim: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

  x_replaced = x.masked_fill(~mask, min_value_of_dtype(x.dtype))
  max_value, max_index = x_replaced.max(dim=dim, keepdim=keepdim)
  return max_value, max_index

def convert_binary_mask_to_infinity_mask(mask: torch.Tensor) -> torch.Tensor:
  return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

class PointerNetwork(nn.Module):
  def __init__(
    self,
    n_hidden: int
  ):
    super().__init__()
    self.n_hidden = n_hidden
    self.w1 = nn.Linear(n_hidden, n_hidden, bias=False)
    self.w2 = nn.Linear(n_hidden, n_hidden, bias=False)
    self.v = nn.Linear(n_hidden, 1, bias=False)

  def forward(
    self,
    x_decoder: torch.Tensor,
    x_encoder: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-16
  ) -> torch.Tensor:
    """
    Args:
      x_decoder: Encoding over the output tokens.
      x_encoder: Encoding over the input tokens.
      mask: Binary mask over the softmax input.
    Shape:
      x_decoder: (B, Ne, C)
      x_encoder: (B, Nd, C)
      mask: (B, Nd, Ne)
    """

    # (B, Nd, Ne, C) <- (B, Ne, C)
    encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(
      -1, x_decoder.shape[1], -1, -1)
    # (B, Nd, 1, C) <- (B, Nd, C)
    decoder_transform = self.w2(x_decoder).unsqueeze(2)
    # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
    prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
    # (B, Nd, Ne) <- (B, Nd, Ne)
    log_score = masked_log_softmax(prod, mask, dim=-1, eps=eps)
    return log_score

class CurveOrderingNet13(nn.Module):
  '''
    no encoder
  '''
  def __init__(
    self,
    c_inputs: int = 5,
    c_embed: int = 8,
    n_heads: int = 2,
    n_layers: int = 1,
    dropout: float = 0.1,
    c_hidden: int = 2,
    internal_attention=False,
  ):
    super().__init__()
    self.c_hidden = c_hidden
    self.c_inputs = c_inputs
    self.c_embed = c_embed
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.dropout = dropout

    self.embedding = nn.Linear(c_inputs, c_embed, bias=False)
    # encoder_layers = nn.TransformerEncoderLayer(c_embed, n_heads, c_hidden, dropout)
    # self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
    if internal_attention:
      decoder_layers = TransformerDecoderLayerWithKnots(c_embed, n_heads, c_hidden, dropout)
      self.decoder = TransformerDecoderWithKnots(decoder_layers, n_layers)
    else:
      decoder_layers = nn.TransformerDecoderLayer(c_embed, n_heads, c_hidden, dropout)
      self.decoder = nn.TransformerDecoder(decoder_layers, n_layers)
    self.pointer = PointerNetwork(n_hidden=c_embed)

    # self.param_decoder=paramDecoder(c_embed,c_hidden,dropout=dropout)
    self.internal_attention=internal_attention

  def forward(
    self,
    batch_data: torch.Tensor,
    point_embeddings:torch.Tensor,
    point_masks:torch.Tensor,
    batch_lengths: torch.Tensor,
    batch_labels: Optional[torch.Tensor] = None,
    knots_embedding: Optional[torch.Tensor] = None,
    knots_mask: Optional[torch.Tensor] = None,
    knots_kv: Optional[torch.Tensor] = None,
    beam_search=True
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if batch_labels is None and beam_search:
      return self.beam_search(batch_data,point_embeddings,point_masks,batch_lengths,knots_embedding,knots_mask,knots_kv=knots_kv,beam_width=1)

    # assumes batch-first inputs
    batch_size = batch_data.shape[0]
    max_seq_len = batch_data.shape[1]
    c_embed = self.c_embed
    n_heads = self.n_heads

    x_embed = self.embedding(batch_data)
    # x_embed=batch_data
    x_embed_norm_first=x_embed.permute(1, 0, 2)
    point_embeddings=point_embeddings.permute(1,0,2)
    # encoder_outputs = self.encoder(x_embed_norm_first)

    if self.internal_attention and knots_embedding is not None:
      knots_embedding=knots_embedding.permute(1,0,2)
      # knots_mask=knots_mask.permute(1,0,2)

    # make mask
    range_tensor = torch.arange(max_seq_len, device=batch_lengths.device,
      dtype=batch_lengths.dtype).expand(batch_size, max_seq_len - len(TOKENS), max_seq_len)
    each_len_tensor = batch_lengths.view(-1, 1, 1).expand(-1, max_seq_len - len(TOKENS), max_seq_len)
    mask_tensor = (range_tensor < each_len_tensor)
    
    mask_tensor2=point_masks.unsqueeze(-2).expand(-1,max_seq_len-len(TOKENS),-1)

    if batch_labels is not None:
      # teacher forcing
      # pass through decoder
      # here memory_mask is (batch_size * n_heads, len_decoder_seq, len_encoder_seq)
      # https://discuss.pytorch.org/t/memory-mask-in-nn-transformer/55230/5
      _bl = torch.cat((torch.zeros_like(batch_labels[:, :1]), batch_labels[:, :-1]), dim=1).permute(1, 0).unsqueeze(-1)
      _bl = _bl.expand(-1, batch_size, c_embed)
      decoder_input = torch.gather(x_embed_norm_first, dim=0, index=_bl)
      decoder_mask = mask_tensor2.repeat((n_heads, 1, 1))
      dm = convert_binary_mask_to_infinity_mask(decoder_mask)

      tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(decoder_input)).to(dm.device)

      if self.internal_attention and knots_embedding is not None:
        knots_mask_tensor = knots_mask.unsqueeze(-2).expand(-1,max_seq_len-len(TOKENS),-1).repeat((n_heads, 1, 1))
        km=convert_binary_mask_to_infinity_mask(knots_mask_tensor)
        decoder_outputs = self.decoder(decoder_input, point_embeddings,knots=knots_embedding,knots_kv=knots_kv,
          knots_mask=km,
          tgt_mask=tgt_mask, memory_mask=dm)
      else:
        decoder_outputs = self.decoder(decoder_input, point_embeddings,
          tgt_mask=tgt_mask, memory_mask=dm)

      # range_tensor = torch.arange(max_seq_len-len(TOKENS), device=batch_lengths.device,
      #   dtype=batch_lengths.dtype).expand(batch_size, -1)
      # each_len_tensor = (batch_lengths-len(TOKENS)).view(-1, 1).expand(-1, max_seq_len - len(TOKENS))
      # tgt_padding_mask = (range_tensor >= each_len_tensor)
      # decoder_outputs = self.decoder(decoder_input, encoder_outputs,
      #   memory_mask=dm,tgt_key_padding_mask=tgt_padding_mask)

      # pass through pointer network
      decoder_outputs=decoder_outputs.permute(1, 0, 2)
      log_pointer_scores = self.pointer(
        decoder_outputs,
        x_embed,
        mask_tensor)
      _, masked_argmaxs = masked_max(log_pointer_scores, mask_tensor, dim=-1)
      # return log_pointer_scores, masked_argmaxs, self.param_decoder(decoder_outputs).squeeze()
      return log_pointer_scores, masked_argmaxs,decoder_outputs
    else:
      log_pointer_scores = []
      masked_argmaxs = []
      encoder_outputs=x_embed_norm_first
      decoder_input = encoder_outputs[:1]
      for _ in range(max_seq_len - len(TOKENS)):
        # pass through decoder network
        decoder_mask = mask_tensor[:, :len(decoder_input)].repeat((n_heads, 1, 1))
        dm = convert_binary_mask_to_infinity_mask(decoder_mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(decoder_input)).to(dm.device)

        if self.internal_attention and knots_embedding is not None:
          knots_mask_tensor = knots_mask.unsqueeze(-2).expand(-1,len(decoder_input),-1).repeat((n_heads, 1, 1))
          km=convert_binary_mask_to_infinity_mask(knots_mask_tensor)
          decoder_outputs = self.decoder(decoder_input, encoder_outputs,knots=knots_embedding,knots_kv=knots_kv,knots_mask=km,
            tgt_mask=tgt_mask, memory_mask=dm)
        else:
          decoder_outputs = self.decoder(decoder_input, encoder_outputs,
            tgt_mask=tgt_mask, memory_mask=dm)
        
        # pass through pointer network
        mask_subset = mask_tensor[:, :len(decoder_outputs)]
        log_pointer_score = self.pointer(
          decoder_outputs.permute(1, 0, 2),
          encoder_outputs.permute(1, 0, 2),
          mask_subset)
        _, masked_argmax = masked_max(log_pointer_score, mask_subset, dim=-1)

        # append new predictions
        log_pointer_scores.append(log_pointer_score[:, -1, :])
        new_maxes = masked_argmax[:, -1]
        masked_argmaxs.append(new_maxes)
        
        # mask out predicted inputs
        # new_max_mask = torch.zeros((mask_tensor.shape[0], mask_tensor.shape[2]),
        #   dtype=torch.bool, device=mask_tensor.device)
        # new_max_mask = new_max_mask.scatter(1, new_maxes.unsqueeze(1), True)
        # new_max_mask[:, :2] = False
        # new_max_mask = new_max_mask.unsqueeze(1).expand(-1, mask_tensor.shape[1], -1)
        # mask_tensor[new_max_mask] = False

        # prepare inputs for next iteration
        next_indices = torch.stack(masked_argmaxs, dim=0).unsqueeze(-1).expand(-1, batch_size, c_embed)
        decoder_input = torch.cat((encoder_outputs[:1], 
          torch.gather(encoder_outputs, dim=0, index=next_indices)), dim=0)
      log_pointer_scores = torch.stack(log_pointer_scores, dim=1)
      masked_argmaxs = torch.stack(masked_argmaxs, dim=1)
      return log_pointer_scores, masked_argmaxs,decoder_outputs.permute(1, 0, 2)

  def beam_search(
    self,
    batch_data: torch.Tensor,
    point_embeddings:torch.Tensor,
    point_masks:torch.Tensor,
    batch_lengths: torch.Tensor,
    knots_embedding: Optional[torch.Tensor] = None,
    knots_mask: Optional[torch.Tensor] = None,
    knots_kv=None,
    beam_width=5,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # assumes batch-first inputs
    batch_size = batch_data.shape[0]
    max_seq_len = batch_data.shape[1]
    c_embed = self.c_embed
    n_heads = self.n_heads

    device=batch_lengths.device
    x_embed = self.embedding(batch_data)
    # x_embed=batch_data
    x_embed_norm_first=x_embed.permute(1, 0, 2)
    point_embeddings=point_embeddings.permute(1,0,2)
    # encoder_outputs = self.encoder(x_embed_norm_first)

    if self.internal_attention and knots_embedding is not None:
      knots_embedding=knots_embedding.permute(1,0,2)
      # knots_mask=knots_mask.permute(1,0,2)

    # make mask
    range_tensor = torch.arange(max_seq_len, device=batch_lengths.device,
      dtype=batch_lengths.dtype).expand(batch_size, max_seq_len - len(TOKENS), max_seq_len)
    each_len_tensor = batch_lengths.view(-1, 1, 1).expand(-1, max_seq_len - len(TOKENS), max_seq_len)
    mask_tensor = (range_tensor < each_len_tensor)

    mask_tensor2=point_masks.unsqueeze(-2).expand(-1,max_seq_len-len(TOKENS),-1)
    # log_pointer_scores = []
    # masked_argmaxs = []
    encoder_outputs=x_embed_norm_first
    decoder_input = encoder_outputs[:1]


    beams = [torch.zeros(batch_size,1,device=device,dtype=torch.long)]
    beam_scores = torch.zeros(batch_size,1,device=device)
    
    # Mask for finished sequences
    # finished = torch.zeros(batch_size, beam_width).bool()
    for _ in range(max_seq_len - len(TOKENS)):
      candidates = []
      candidate_scores = []

      for i,beam in enumerate(beams):
      # pass through decoder network
        if _>0:
          decoder_input = torch.gather(encoder_outputs, dim=0, index=beam.permute(1,0).unsqueeze(-1).expand(-1,-1,encoder_outputs.shape[-1]))

        decoder_mask = mask_tensor2[:, :len(decoder_input)].repeat((n_heads, 1, 1))
        dm = convert_binary_mask_to_infinity_mask(decoder_mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(decoder_input)).to(dm.device)

        if self.internal_attention and knots_embedding is not None:
          knots_mask_tensor = knots_mask.unsqueeze(-2).expand(-1,len(decoder_input),-1).repeat((n_heads, 1, 1))
          km=convert_binary_mask_to_infinity_mask(knots_mask_tensor)
          decoder_outputs = self.decoder(decoder_input, point_embeddings,knots=knots_embedding,knots_kv=knots_kv,knots_mask=km,
            tgt_mask=tgt_mask, memory_mask=dm)
        else:
          decoder_outputs = self.decoder(decoder_input, point_embeddings,
            tgt_mask=tgt_mask, memory_mask=dm)

        # (B,Ne) 
        new_max_mask = torch.zeros((mask_tensor.shape[0], mask_tensor.shape[2]),
          dtype=torch.bool, device=mask_tensor.device)

        # (B,Ne) 
        new_max_mask = new_max_mask.scatter(1, beam, True)

        # (B,Nd,Ne) 
        new_max_mask = new_max_mask.unsqueeze(1).expand(-1, len(decoder_outputs), -1)
        mask_subset=torch.masked_fill(mask_tensor[:,:len(decoder_outputs)],new_max_mask,False)
        # mask_tensor[new_max_mask] = False

        # pass through pointer network
        # mask_subset = mask_tensor[:, :len(decoder_outputs)]
        log_pointer_score = self.pointer(
          decoder_outputs.permute(1, 0, 2),
          encoder_outputs.permute(1, 0, 2),
          mask_subset)
        # _, masked_argmax = masked_max(log_pointer_score, mask_subset, dim=-1)

        last_scores = log_pointer_score[:, -1, :]
        topk_probs,topk_indices=torch.topk(last_scores, beam_width, dim=-1)

        end_mask=_+1>=batch_lengths
        end_mask_expanded=end_mask.unsqueeze(-1).expand(-1,beam_width).contiguous()
        end_mask_expanded[:,0]=False
        topk_probs=torch.masked_fill(topk_probs,end_mask_expanded,value=min_value_of_dtype(topk_probs.dtype))

        for j in range(beam_width):
          candidates.append(torch.cat([beam,topk_indices[:, j].unsqueeze(-1)], dim=-1))
          candidate_scores.append(beam_scores[i]+topk_probs[:,j])
        # candidate_scores.append(beam_scores[:,i:i+1]+topk_probs)

      # (Brances, B, seq_len)
      candidates=torch.stack(candidates,dim=0)

      # (Brances, B)
      candidate_scores=torch.stack(candidate_scores,dim=0)

      # (beam width, B)
      # (beam width, B)
      topk_scores,topk_score_indices=candidate_scores.topk(beam_width,dim=0)

      # (beam width, B, seq_len)
      beams=torch.gather(candidates,dim=0,index=topk_score_indices.unsqueeze(-1).expand(-1,-1,candidates.shape[-1]))
      beam_scores=topk_scores

    best_score=beam_scores[0]
    best_indices=beams[0,:,1:]

    return best_score, best_indices, decoder_outputs.permute(1, 0, 2)