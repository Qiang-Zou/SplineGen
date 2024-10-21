from util import nurbs_eval
from typing import Tuple, Union, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
# from shapely import geometry

import models.onn
import models.pointsEncoder
import models.encoder_decoder
import models.param_decoder
import models.additioanl_layer
import models.kkn
import models
TOKENS = {
  '<eos>': 0
}

def getModel_Simple(device='cuda',encoder_load_path=None,model_load_path=None,input_dim=3):
     d_model = 512  # Embedding dimension
     nhead = 4

     num_decoder_layers = 3
     dim_feedforward = 2048
     dropout = 0.01

     en_input_dim=input_dim
     en_num_encoder_layers=3
     en_nhead=4
     en_dropout=0.05
     points_encoder=models.pointsEncoder.PointsEncoder(
          en_input_dim, 
          hidden_dim=d_model, 
          num_layers=en_num_encoder_layers, 
          num_head=en_nhead, 
          dim_feedforward=dim_feedforward,
          dropout=en_dropout).to(device=device)

     decoder=models.kkn.KPN2(
          d_model,
          num_decoder_layers=num_decoder_layers,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout).to(device=device)

     model=models.encoder_decoder.PointsEncoderDecoder(points_encoder,decoder,lock_encoder=True).to(device)

     if model_load_path:
          model.load_state_dict(torch.load(model_load_path))
          print("Whole model loaded")
     else:
          if encoder_load_path:
               states=torch.load(encoder_load_path)
               new_states={}
               for key in states:
                    if key.startswith('encoder'):
                         new_states[key[8:]]=states[key]

               model.encoder.load_state_dict(new_states)
               print("Encoder loaded")

               #   new_states={}
               #   for key in states:
               #       if key.startswith('decoder1'):
               #           new_states[key[len('decoder1.'):]]=states[key]

               #   model.decoder.load_state_dict(new_states)
               #   print("Decoder loaded")
     return model

def getModel_SimpleEncoder_Knots(device='cuda',input_dim=3,internal_attention=True,model_load_path='',knot_load_path=''):
     c_embed = 512 # 64# 16
     c_hidden2 = 1024 # 16
     n_heads = 4
     n_layers = 3

     dropout = 0.05

     torch.random.manual_seed(231)
     # min_samples=20``

     en_input_dim=input_dim
     d_model=512 
     en_num_encoder_layers=3
     en_nhead=4
     dim_feedforward=2048
     en_dropout=0.05
     points_encoder=models.pointsEncoder.PointsEncoder(
          en_input_dim, 
          hidden_dim=d_model, 
          num_layers=en_num_encoder_layers, 
          num_head=en_nhead, 
          dim_feedforward=dim_feedforward,
          dropout=en_dropout).to(device=device)

     points_encoder2=models.pointsEncoder.PointsEncoder(
          en_input_dim, 
          hidden_dim=d_model, 
          num_layers=en_num_encoder_layers, 
          num_head=en_nhead, 
          dim_feedforward=dim_feedforward,
          dropout=en_dropout).to(device=device)
     k_nhead = 4
     k_num_decoder_layers = 3
     k_dropout = 0.01
     # Create DataLoader for training data
     print('# Create DataLoader for training data')

     decoder_k=models.kkn.KPN2(
          d_model,
          num_decoder_layers=k_num_decoder_layers,
          nhead=k_nhead,
          dim_feedforward=dim_feedforward,
          dropout=k_dropout).to(device=device)

     new_model = models.encoder_decoder.PointsEncoderDecoder11(
          points_encoder,
          encoder2=points_encoder2,
          knot_decoder=decoder_k,
          param_decoder=models.onn.CurveOrderingNet13(c_inputs=c_embed+len(TOKENS),c_embed=c_embed, n_heads=n_heads,
               n_layers=n_layers, dropout=dropout, c_hidden=c_hidden2,internal_attention=internal_attention),
          param_decoder2=models.param_decoder.paramDecoderMLP(c_input=c_embed,hidden_dim=c_hidden2,dropout=dropout)
          ).to(device)

     new_model.lock_encoder=True
     new_model.lock_knots=True
     model=new_model

     if model_load_path:
          model.load_state_dict(torch.load(model_load_path))
          print("Whole model loaded")
     else:
          if knot_load_path:
               states=torch.load(knot_load_path)
               new_states={}
               for key in states:
                    if key.startswith('encoder'):
                         new_states[key[len('encoder.'):]]=states[key]

               points_encoder.load_state_dict(new_states)
               print("Encoder loaded")

               new_states={}
               for key in states:
                    if key.startswith('decoder'):
                         new_states[key[len('decoder.'):]]=states[key]

               decoder_k.load_state_dict(new_states)
               print("Knot Decoder loaded")

     return model

def getSplineGen(device='cuda',model_load_path='',base_model_load_path='',input_dim=3):

     base_model=getModel_SimpleEncoder_Knots(device=device,model_load_path=base_model_load_path,input_dim=input_dim) 

     additional_model=models.additioanl_layer.AdditionalLayer(3).to(device)

     model=models.encoder_decoder.SplineGen(base_model,additional_model)

     if model_load_path:
          model.load_state_dict(torch.load(model_load_path))
          print("Whole model loaded")

     return model
