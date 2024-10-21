import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple
from util import curve

KNOT_TOKENS = {'<sos>':0,'<eos>':1}

class CurveDataset_for_encoder(Dataset):
  def __init__(
    self,
    data_path,
  ):
    print('data loading...')
    data=np.load(data_path)

    degree=data['degree']
    self.degree=degree

    print('degree:',degree)
    self.keys=[]

    points=data['points'] 
    num_curve,max_point_len,dimension=points.shape
    self.dimension=dimension
    print('points shape:',points.shape)
    
    points_len_array=data['points_len'] # (num_curve)
    
    self.points=points
    self.params=data['params']
    self.points_mask=self.getPaddingMask(max_point_len,points_len_array)
    self.params_expanded,self.params_mask_expanded=self.add_tokens(self.params,self.points_mask)

    self.keys.extend(['params_expanded','params_mask_expanded'])
    self.keys.extend(['points','params','points_mask'])

    ctrl_pts=data['ctrl_pts']
    _,max_ctrl_len,__=ctrl_pts.shape
    print('ctrl_pts shape:',ctrl_pts.shape)
    ctrl_pts_len_array=data['ctrl_pts_len'] # (num_curve)
    
    self.knots=data['knots']
    self.knots_mask=self.getPaddingMask(max_ctrl_len+degree+1,ctrl_pts_len_array+(degree+1))
    self.knots_expanded,self.knots_mask_expanded=self.add_tokens(self.knots,self.knots_mask)
    self.keys.extend(['knots','knots_mask','knots_expanded','knots_mask_expanded'])
    
  def getPaddingMask(self,max_len,len_array):
      range_array=np.tile(np.arange(max_len),(len(len_array),1))
      each_len= np.broadcast_to(np.expand_dims(len_array,axis=-1),(len(len_array),max_len))     
      
      mask=range_array<each_len
      
      return mask
    
  def add_tokens(self, knots,mask):
    knots=torch.tensor(knots)
    mask=torch.tensor(mask)
    # 1. Add the SOS and EOS tokens to the points_params
    # points_params = torch.cat((torch.full_like(points_params[:, :1], SOS), points_params, torch.full_like(points_params[:, :1], EOS)), dim=1)


    # 2. Add two extra channels to points_params to indicate the position of SOS and EOS
    # No

    # 3. Add SOS token at the start of knots and EOS token after the first non-zero element from the end
    # Also, add two extra dimensions to indicate the positions of SOS and EOS
    batch_size, length = knots.size()
    new_knots = torch.zeros((batch_size, length + len(KNOT_TOKENS), 1+len(KNOT_TOKENS)))

    
    new_knots[:,1:length+1,0] = knots
    new_knots[:,0,1] = 1.0 # start token

    batch_knots_mask=mask
    batch_knots_mask_new = torch.zeros(batch_knots_mask.size(0), batch_knots_mask.size(1) + len(KNOT_TOKENS), dtype=bool)
    batch_knots_mask_new[:,0] = True
    batch_knots_mask_new[:, 1:batch_knots_mask.size(1) + 1] = batch_knots_mask

    new_knots[...,2].masked_fill_(~batch_knots_mask_new,1.0)

    return new_knots,batch_knots_mask_new

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data={}
    for key in self.keys:

        v=getattr(self,key)[idx]
        if v.dtype==np.float64:
          data[key]=np.asarray(v,dtype=np.float32)
        else:
          data[key]=v
  
    return data

  def __len__(self) -> int:
    return len(self.points)

  def to(self,device):
    for key in self.keys:
      setattr(self,key,getattr(self,key).to(device))
    return self

  def __len__(self) -> int:
    return len(self.points)
    