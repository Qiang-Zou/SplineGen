from util import nurbs_eval
from typing import Tuple, Union, Optional
import os
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split

from torch.utils.tensorboard import SummaryWriter
from dataset.curveDataset import CurveDataset
import train.getModel as getModel
from util import AverageMeter,masked_accuracy

TRAIN_WEIGHTS=[0.1,0.90,0]

TOKENS = {
  '<eos>': 0
}
  
def train(data_path,log_dir,model_save_path,base_model_load_path,use_cuda=True,n_workers=4,n_epochs=1000,batch_size=256,lr=1e-6):
    model_save_path=model_save_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    torch.random.manual_seed(231)
    
    dataset=CurveDataset(data_path,use_points_params=True,use_knots=True,use_orders=True,
                          random_select_rate=None)
    train_dataset,val_dataset=random_split(dataset=dataset,lengths=(0.8,0.2))

    print(f'# val:   {len(dataset):7d}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
      num_workers=n_workers,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
      num_workers=n_workers,shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    model=getModel.getSplineGen(device=device,base_model_load_path=base_model_load_path,input_dim=dataset.dimension)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def knots_loss_fn(knots_pred, knots, knots_mask):
        loss = torch.nn.functional.mse_loss(knots_pred, knots, reduction='none')
        
        loss = loss.sum(dim=-1)
        loss = loss.masked_fill(knots_mask == 0, 0)
        loss = loss.sum(dim=-1)
        knots_len = knots_mask.sum(dim = -1)

        loss = loss/knots_len
        return torch.mean(loss)

    def params_loss_fn(params_label,params,params_mask):
        loss = torch.nn.functional.mse_loss(params, params_label, reduction='none')
    
        loss = loss.masked_fill(params_mask == 0, 0)
        loss = loss.sum(dim=-1)
        # loss = loss.sum(dim=-1)
        valid_len = params_mask.sum(dim = -1)

        loss = loss/valid_len
        loss= torch.mean(loss)

        return loss
    def criterion(score,label,params,params_label,params_mask,knots,knots_pred,knots_mask,w=TRAIN_WEIGHTS):
        loss1=torch.nn.NLLLoss(ignore_index=TOKENS['<eos>'])(score,label)

        loss2=params_loss_fn(params,params_label,params_mask)

        loss3=knots_loss_fn(knots_pred,knots,knots_mask)
        return loss1,loss2,loss3,w[0]*loss1+w[1]*loss2+w[2]*loss3

    train_loss = AverageMeter()
    train_loss_order = AverageMeter()
    train_loss_param = AverageMeter()
    train_loss_knots=AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_loss_order=AverageMeter()
    val_loss_param=AverageMeter()
    val_loss_knots=AverageMeter()
    val_accuracy = AverageMeter()

# begin training
    degree=3
    cache_t=[]
    cache_v=[]
    for epoch in range(n_epochs):
        model.train()
        cache=cache_t
        for bat, input in enumerate(tqdm(train_loader)):
            batch_labels = input['targets'].to(device)
            batch_lengths = input['length'].to(device)
            batch_mask=input['points_mask'].to(device)
            batch_params=input['params'].to(device=device,dtype=torch.float32)
            batch_points=input['points'].to(device)
            batch_knots=input['knots_expanded'].to(device)
            batch_knots_mask=input['knots_mask_expanded'].to(device)

            optimizer.zero_grad()

            knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                batch_points,batch_params,batch_mask, batch_lengths,
                batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
            )


            knot_length=torch.sum(knots_mask,dim=-1)-1

            loss,ctrl=nurbs_eval.getCtrlPts(degree,[params,batch_points,batch_mask,knots,knot_length])
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item()/ batch_knots.size(0), batch_knots.size(0))
            mask = batch_labels != TOKENS['<eos>']
            acc = masked_accuracy(pointer_argmaxs, batch_labels, mask).item()
            train_accuracy.update(acc, mask.int().sum().item())

        model.eval()

        cache=cache_v
        with torch.no_grad():
          for bat, input in enumerate(tqdm(val_loader)):
              batch_labels = input['targets'].to(device)
              batch_lengths = input['length'].to(device)
              batch_mask=input['points_mask'].to(device)
              batch_params=input['params'].to(device=device,dtype=torch.float32)
              batch_points=input['points'].to(device)
              batch_knots=input['knots_expanded'].to(device)
              batch_knots_mask=input['knots_mask_expanded'].to(device)

              # with torch.no_grad():
              #   knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = new_model(
              #       batch_points,batch_params,batch_mask, batch_lengths,
              #       batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
              #   )
                  # cache.append({'knots':knots,'knots_mask':knots_mask,'params':params,'pointer_argmaxs':pointer_argmaxs})
              # params,knots=model(batch_points,params,batch_mask,knots,knots_mask)

              knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                  batch_points,batch_params,batch_mask, batch_lengths,
                  batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
              )
              # order_loss,param_loss,knots_loss,loss = criterion(
              #   log_pointer_scores.view(-1, log_pointer_scores.shape[-1]),
              #   batch_labels.reshape(-1),
              #   params=params,params_label=batch_params,params_mask=batch_mask,
              #   knots=knots,knots_pred=batch_knots[:,1:],knots_mask=batch_knots_mask[:,:-1])

              knot_length=torch.sum(knots_mask,dim=-1)-1
              # indices=pointer_argmaxs.unsqueeze(-1).expand(-1, -1, 3).clip(0,batch_points.size(1)-1).long()
              # sorted_points=torch.gather(batch_points,1,indices)
              # params=chordPPN.chordPPN(centripetal=False)(sorted_points,points_mask=batch_mask)
              loss,ctrl=nurbs_eval.getCtrlPts(degree,[params,batch_points,batch_mask,knots,knot_length])


              val_loss.update(loss.item()/ batch_knots.size(0), batch_knots.size(0))
              mask = batch_labels != TOKENS['<eos>']
              acc = masked_accuracy(pointer_argmaxs, batch_labels, mask).item()
              val_accuracy.update(acc, mask.int().sum().item())

        if (epoch + 1) % 5 == 0:
            writer.flush()
            # save model every 10 epoch
            torch.save(model.state_dict(), model_save_path+f'/epoch_{epoch+1}'+'.pth')
                  
        writer.add_scalar('Train/Loss',train_loss.avg,epoch+1)
        writer.add_scalar('Train/Accuracy',train_accuracy.avg,epoch+1)
        writer.add_scalar('Val/Loss',val_loss.avg,epoch+1)
        writer.add_scalar('Val/Accuracy',val_accuracy.avg,epoch+1)

        print('Loss:',val_loss.avg,'epoch:', epoch+1)
        print('Accuracy:',val_accuracy.avg,'epoch:',epoch+1)

        train_loss.reset()
        train_accuracy.reset()
        train_loss_param.reset()
        train_loss_order.reset()
        val_loss.reset()
        val_accuracy.reset()
        val_loss_order.reset()
        val_loss_param.reset()
                  

if __name__=='__main__':
    train()