'''
  in this version, the decoder is also used to predict the param
'''
import os
import datetime
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader,random_split

from torch.utils.tensorboard import SummaryWriter
from dataset.curveDataset import CurveDataset
import train.getModel as getModel
from util import AverageMeter,masked_accuracy

TOKENS = {
  '<eos>': 0
}

def train(data_path,model_save_path,log_path,knot_model_load_path,train_weights=[0.1,0.9],use_cuda=True,
          n_workers=4,n_epochs=1000,batch_size=256,lr=1e-4,save_epoch=5):
    torch.random.manual_seed(231)

    use_cuda = True

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(log_path):
      os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_save_path=model_save_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    dataset=CurveDataset(data_path,use_points_params=True,use_knots=True,use_orders=True,
                          random_select_rate=None)
    
    input_dim=dataset.dimension

    train_dataset,val_dataset=random_split(dataset=dataset,lengths=(0.8,0.2))

    print(f'# train: {len(train_dataset):7d}')
    print(f'# val:   {len(val_dataset):7d}')


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
      num_workers=n_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
      num_workers=n_workers,shuffle=False)

    model=getModel.getModel_SimpleEncoder_Knots(device=device,knot_load_path=knot_model_load_path,input_dim=input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    def criterion(score,label,params,params_label,params_mask,w):
        loss1=torch.nn.NLLLoss(ignore_index=TOKENS['<eos>'])(score,label)

        loss2=params_loss_fn(params,params_label,params_mask)

        return loss1,loss2,w[0]*loss1+w[1]*loss2

    train_loss = AverageMeter()
    train_loss_order = AverageMeter()
    train_loss_param = AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_loss_order=AverageMeter()
    val_loss_param=AverageMeter()
    val_accuracy = AverageMeter()

# begin training
    for epoch in range(n_epochs):
        model.train()
        print(f'Epoch {epoch} training...')
        for bat, input in enumerate(tqdm(train_loader)):
        # for bat, (batch_data, batch_labels, batch_lengths,batch_mask,batch_params,batch_points) in enumerate(tqdm(train_loader)):
            batch_labels = input['targets'].to(device)
            batch_lengths = input['length'].to(device)
            batch_mask=input['points_mask'].to(device)
            batch_params=input['params'].to(device=device,dtype=torch.float32)
            batch_points=input['points'].to(device)
            batch_knots=input['knots_expanded'].to(device)
            batch_knots_mask=input['knots_mask_expanded'].to(device)

            optimizer.zero_grad()
            # knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
            #     batch_points,batch_mask, batch_lengths,
            #     batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
            # )
            knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                batch_points,batch_params,batch_mask, batch_lengths,
                batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1]
            )

            order_loss,param_loss,loss = criterion(
               log_pointer_scores.view(-1, log_pointer_scores.shape[-1]),
               batch_labels.reshape(-1),
               params=params,params_label=batch_params,params_mask=batch_mask,w=train_weights
               )

            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), batch_knots.size(0))
            mask = batch_labels != TOKENS['<eos>']
            acc = masked_accuracy(pointer_argmaxs, batch_labels, mask).item()
            train_accuracy.update(acc, mask.int().sum().item())
            train_loss_order.update(order_loss.item(), batch_knots.size(0))
            train_loss_param.update(param_loss.item(), batch_mask.int().sum().item())
            # train_loss_knots.update(knots_loss.item(), batch_knots_mask[:,1:].int().sum().item())

        # if bat % log_interval == 0:
        #   print(f'Epoch {epoch}: '
        #         f'Train [{bat * len(batch_data):9d}/{len(train_dataset):9d} '
        #         f'Loss: {train_loss.avg:.6f}\tAccuracy: {train_accuracy.avg:3.4%}')
                  
        writer.add_scalar('Loss/train',train_loss.avg,epoch)
        writer.add_scalar('Order Loss/train',train_loss_order.avg,epoch)
        writer.add_scalar('Param Loss/train',train_loss_param.avg,epoch)
        writer.add_scalar('Accuracy/train',train_accuracy.avg,epoch)

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
                  batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1]
              )

              order_loss,param_loss,loss = criterion(
                log_pointer_scores.view(-1, log_pointer_scores.shape[-1]),
                batch_labels.reshape(-1),
                params=params,params_label=batch_params,params_mask=batch_mask,w=train_weights)

              val_loss.update(loss.item(), batch_knots.size(0))
              mask = batch_labels != TOKENS['<eos>']
              acc = masked_accuracy(pointer_argmaxs, batch_labels, mask).item()
              val_accuracy.update(acc, mask.int().sum().item())
              val_loss_order.update(order_loss.item(), batch_knots.size(0))
              val_loss_param.update(param_loss.item(), batch_mask.int().sum().item())

        # if bat % log_interval == 0:
        #   print(f'Epoch {epoch}: '
        #         f'Train [{bat * len(batch_data):9d}/{len(train_dataset):9d} '
        #         f'Loss: {train_loss.avg:.6f}\tAccuracy: {train_accuracy.avg:3.4%}')
                  
        writer.add_scalar('Loss/val',val_loss.avg,epoch)
        writer.add_scalar('Order Loss/val',val_loss_order.avg,epoch)
        writer.add_scalar('Param Loss/val',val_loss_param.avg,epoch)
        writer.add_scalar('Accuracy/val',val_accuracy.avg,epoch)
        # writer.add_scalar('Accuracy/val',train_accuracy.avg,epoch)

        # print(f'Epoch {epoch}: Val\tLoss: {val_loss.avg:.6f} '
        #         f'\tAccuracy: {val_accuracy.avg:3.4%} '
        if (epoch + 1) % save_epoch == 0:
            writer.flush()
            # save model every 10 epoch
            torch.save(model.state_dict(), model_save_path+f'_epoch_{epoch}'+'.pth')
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
    train()