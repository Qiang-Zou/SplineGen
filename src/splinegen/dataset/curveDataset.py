import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple
from util import curve

TOKENS = {'<eos>':0}
KNOT_TOKENS = {'<sos>':0,'<eos>':1}

class CurveDataset(Dataset):
  def __init__(
    self,
    data_path,
    use_points_params=True,
    use_knots=True,
    use_ctrl_pts=True,
    use_orders=False,
    random_select_rate=None
  ):
    print('data loading...')
    data=np.load(data_path)

    degree=data['degree']
    self.degree=degree

    print('degree:',degree)
    self.keys=[]

    if use_points_params:
        points=data['points'] 
        num_curve,max_point_len,dimension=points.shape
        print('points shape:',points.shape)
        
        points_len_array=data['points_len'] # (num_curve)
        
        self.points=points
        self.params=data['params']
        self.points_mask=self.getPaddingMask(max_point_len,points_len_array)
        self.points_len=points_len_array
        
        self.keys.extend(['points','params','points_mask','points_len'])

        self.use_orders=use_orders
        self.dimension=dimension
        if use_orders:
          self.max_samples=max_point_len

          self.points_len_array=points_len_array # (num_curve,)
          
          self.targets=[np.arange(l) for l in self.points_len_array]

    if use_ctrl_pts or use_knots: 
        ctrl_pts=data['ctrl_pts']
        _,max_ctrl_len,__=ctrl_pts.shape
        print('ctrl_pts shape:',ctrl_pts.shape)
        ctrl_pts_len_array=data['ctrl_pts_len'] # (num_curve)
        
        if use_ctrl_pts:
            self.ctrl_pts=data['ctrl_pts']
            self.ctrl_mask=self.getPaddingMask(max_ctrl_len,ctrl_pts_len_array)
            
            self.keys.extend(['ctrl_pts','ctrl_mask'])

        if use_knots:
            self.knots=data['knots']
            self.knots_mask=self.getPaddingMask(max_ctrl_len+degree+1,ctrl_pts_len_array+(degree+1))
            self.knots_expanded,self.knots_mask_expanded=self.add_tokens(self.knots,self.knots_mask)
            self.keys.extend(['knots','knots_mask','knots_expanded','knots_mask_expanded'])

    self.random_select_rate=random_select_rate
    
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

    if self.random_select_rate is not None:
      reserved_num=int(data['points_len']*(1-self.random_select_rate))

      select=np.random.permutation(int(data['points_len']))[:reserved_num]
      select=np.sort(select)
      # target=np.random.permutation(reserved_num)
      data['points'][:reserved_num]=data['points'][select]
      # data['params'][target]=data['params'][:reserved_num]
      data['params'][:reserved_num]=data['params'][select]
      data['points_mask'][reserved_num:]=False
      data['points_len']=reserved_num
      # data['targets'][:reserved_num]=target
      # target=data['targets']
      target=np.arange(reserved_num)
    else:
      if self.use_orders:
        target=self.targets[idx]
  
    if self.use_orders:
      points, targets, length = self.pad_item(data['points'],target)
      data['points_']=points.float()
      data['targets']=targets
      data['length']=length

    return data

  def __len__(self) -> int:
    return len(self.points)

  def pad_item(
    self,
    points: list,
    targets: list,
  ) -> Tuple[torch.tensor, torch.Tensor]:
    n_tokens = len(TOKENS)
    
    # points_padded = np.zeros((self.max_samples + n_tokens, 3 + n_tokens),
    points_padded = np.zeros((self.max_samples + n_tokens, self.dimension + n_tokens),
      dtype=np.float32)
    targets_padded = np.ones((self.max_samples), dtype=np.int64) \
      * TOKENS['<eos>']
                               
    len_points=len(targets)
    
    # points_padded[TOKENS['<sos>'], 2] = 1.0
    # points_padded[TOKENS['<eos>'], 3] = 1.0
    points_padded[TOKENS['<eos>'], self.dimension] = 1.0
    points_padded[n_tokens:n_tokens + len_points, :self.dimension] = points[:len_points]
    # points_padded[n_tokens + len(points):, 4] = 1.0
    targets_padded[:len(targets)] = np.array([t + n_tokens for t in targets])

    points_padded = torch.tensor(points_padded, dtype=torch.float32)
    targets_padded = torch.tensor(targets_padded, dtype=torch.int64)
    length = torch.tensor(len_points + 2, dtype=torch.int64)
    return points_padded, targets_padded, length

  def getCurve(self, idx: int)->curve.BSplineCurve:
    ctrl_len=sum(self.ctrl_mask)
    ctrl_pts=self.ctrl_pts[idx,:ctrl_len]
    
    knot_len=sum(self.knots_mask)
    knots=self.knots[idx,:knot_len]

    degree=self.degree
    
    return curve.BSplineCurve(degree=degree,control_pts=ctrl_pts,knot_vector=knots)
    

  def to(self,device):
    for key in self.keys:
      setattr(self,key,getattr(self,key).to(device))
    return self

  def __len__(self) -> int:
    return len(self.points)
    