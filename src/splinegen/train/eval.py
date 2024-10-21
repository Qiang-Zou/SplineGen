'''
  in this version, the decoder is also used to predict the param
'''
from typing import Tuple, Union, Optional
from util.curveSampling2D import EqualChordLenSamplingParams,NoisedEqualChordLenSamplingParamsMulti,NoisedEqualChordLenSamplingParamsCpp,curveGradientVariation
import util.approximation as approximation
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,random_split
# from shapely import geometry

from dataset.curveDataset import CurveDataset
from util import nurbs_eval
import train.getModel as getModel
from util import masked_accuracy,AverageMeter

TOKENS = {
  '<eos>': 0
}

def eval(data_path,model_load_path,use_cuda=True,n_workers=4,batch_size=256):
    degree=3
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    torch.random.manual_seed(231)
    dataset=CurveDataset(data_path,use_points_params=True,use_knots=True,use_orders=True,
                          random_select_rate=None)

    # train_dataset,val_dataset,test_set=random_split(dataset=dataset,lengths=(0.8,0.1,0.1))
    test_set=dataset

    val_loader = DataLoader(test_set, batch_size=batch_size,
      num_workers=n_workers,shuffle=False)


    model=getModel.getSplineGen(device=device,model_load_path=model_load_path)

    INPUT_ORDERED=False

    def params_loss_fn(params_label,params,params_mask):
        loss = torch.nn.functional.mse_loss(params, params_label, reduction='none')
    
        # inverted_Token_mask = 1 - Token_mask
        # masked_loss = loss * inverted_Token_mask
        
        loss = loss.masked_fill(params_mask == 0, 0)
        loss = loss.sum(dim=-1)
        # loss = loss.sum(dim=-1)
        valid_len = params_mask.sum(dim = -1)

        loss = loss/valid_len
        loss= torch.mean(loss)

        return loss

    train_loss = AverageMeter()
    train_loss_order = AverageMeter()
    train_loss_param = AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_loss2 = AverageMeter()
    val_loss3=AverageMeter()
    val_loss_order=AverageMeter()
    val_loss_param=AverageMeter()
    val_accuracy = AverageMeter()

# begin training
    if not INPUT_ORDERED:
      for epoch in range(1):
          print(f'Epoch {epoch} validating...')
          model.eval()

          with torch.no_grad():
            for bat, input in enumerate(tqdm(val_loader)):
                batch_labels = input['targets'].to(device)
                batch_lengths = input['length'].to(device)
                batch_mask=input['points_mask'].to(device)
                batch_params=input['params'].to(device=device,dtype=torch.float32)
                batch_points=input['points'].to(device)
                batch_knots=input['knots_expanded'].to(device)
                batch_knots_mask=input['knots_mask_expanded'].to(device)

                # knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                #     batch_points,batch_mask, batch_lengths,
                #     batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
                # )
                knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                    batch_points,batch_params,batch_mask, batch_lengths,
                    batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],eval=True
                )


                point_length=torch.sum(batch_mask,dim=-1)
                knot_length=torch.sum(knots_mask,dim=-1)-1

                loss1,loss2,loss3,ctrl=nurbs_eval.getCtrlPts2(degree,[params,batch_points,batch_mask,knots,knot_length])


                val_loss.update(loss1.item()/ batch_lengths.size(0), batch_lengths.size(0))
                val_loss2.update(loss2.item()/ batch_lengths.size(0), batch_lengths.size(0))
                val_loss3.update(loss3.item()/ batch_lengths.size(0), batch_lengths.size(0))
                    
          # writer.add_scalar('Loss/val',val_loss.avg,epoch)
          print('accuracy: ',val_accuracy.avg)
          print('loss: ',val_loss.avg)
          print('loss_sum: ',val_loss2.avg)
          print('hausdoff loss: ',val_loss3.avg)
          # writer.add_scalar('Accuracy/val',train_accuracy.avg,epoch)

          # print(f'Epoch {epoch}: Val\tLoss: {val_loss.avg:.6f} '
          #         f'\tAccuracy: {val_accuracy.avg:3.4%} '
          #         )
          train_loss.reset()
          train_accuracy.reset()
          train_loss_param.reset()
          train_loss_order.reset()
          val_loss.reset()
          val_accuracy.reset()
          val_loss_order.reset()
          val_loss_param.reset()
    else:
      for epoch in range(1):
          print(f'Epoch {epoch} validating...')
          model.eval()

          with torch.no_grad():
            for bat, input in enumerate(tqdm(val_loader)):
                batch_labels = input['targets'].to(device)
                batch_lengths = input['length'].to(device)
                batch_mask=input['points_mask'].to(device)
                batch_params=input['params'].to(device=device,dtype=torch.float32)
                batch_points=input['points'].to(device)
                batch_knots=input['knots_expanded'].to(device)
                batch_knots_mask=input['knots_mask_expanded'].to(device)

                knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                    batch_points,batch_params,batch_mask, batch_lengths,
                    batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
                )


                point_length=torch.sum(batch_mask,dim=-1)
                knot_length=torch.sum(knots_mask,dim=-1)-1

                loss1,loss2,loss3,ctrl=nurbs_eval.getCtrlPts2(degree,[params,batch_points,batch_mask,knots,knot_length])

                mask = batch_labels != TOKENS['<eos>']
                acc = masked_accuracy(pointer_argmaxs, batch_labels, mask).item()
                val_accuracy.update(acc, mask.int().sum().item())

                val_loss.update(loss1.item()/ batch_lengths.size(0), batch_lengths.size(0))
                    
          # writer.add_scalar('Loss/val',val_loss.avg,epoch)
          print('accuracy: ',val_accuracy.avg)
          print('loss: ',val_loss.avg)
          print('loss_sum: ',val_loss2.avg)
          print('hausdoff loss: ',val_loss3.avg)
          # writer.add_scalar('Accuracy/val',train_accuracy.avg,epoch)

          # print(f'Epoch {epoch}: Val\tLoss: {val_loss.avg:.6f} '
          #         f'\tAccuracy: {val_accuracy.avg:3.4%} '
          #         )
          train_loss.reset()
          train_accuracy.reset()
          train_loss_param.reset()
          train_loss_order.reset()
          val_loss.reset()
          val_accuracy.reset()
          val_loss_order.reset()
          val_loss_param.reset()
                  

if __name__=='__main__':
    eval()