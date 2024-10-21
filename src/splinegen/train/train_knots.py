from util.statistic import AverageMeter
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import re
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
import datetime
from dataset.curveDataset import CurveDataset
from torch.utils.data import random_split
import train.getModel as getModel

# if __name__ == '__main__':
# 
def train(data_path,log_path,encoder_path,knot_model_save_dir):
    dataset = CurveDataset(
        data_path,
        random_select_rate=None
        )
    log_dir=log_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = knot_model_save_dir+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    device='cuda'
    p = 3
    input_dim=dataset.dimension
    learning_rate = 0.0001 #This is very good: learning_rate = 0.0001
    # learning_rate = 0.00001 # have a test
    epochs = 1000
    batch_size =  255
    # Create DataLoader for training data
    print('# Create DataLoader for training data')

    train_dataset,val_dataset=random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)],generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model=getModel.getModel_Simple(device=device,encoder_load_path=encoder_path,input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_fn(knots_pred, knots, knots_mask):
        loss = torch.nn.functional.mse_loss(knots_pred, knots, reduction='none')
        
        loss = loss.sum(dim=-1)
        loss = loss.masked_fill(knots_mask == 0, 0)
        loss = loss.sum(dim=-1)
        knots_len = knots_mask.sum(dim = -1)

        loss = loss/knots_len
        return torch.mean(loss)
    criterion = loss_fn  # Mean Squared Error Loss

    # Define the directory where the models are saved
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def train_step(model,batch,loss_avg:AverageMeter,criterion,train=True):
        batch_points = batch['points'].to(device)
        batch_points_mask = batch['points_mask'].to(device)
        batch_knots = batch['knots_expanded'].to(device)
        batch_knots_mask = batch['knots_mask_expanded'].to(device)

        output,_,_=model(batch_points,batch_points_mask,batch_knots[:,:-1],batch_knots_mask[:,:-1])
        loss = criterion(output,batch_knots[:,1:],batch_knots_mask[:,:-1])

        loss_avg.update(loss.item()*batch_points.size(0),batch_points.size(0))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_avg

    # Training loop
    for epoch in range(epochs):

        train_loss_recoder=AverageMeter()
        val_loss_recoder=AverageMeter()

        model.train()
        train_loss_total = 0
        for batch in tqdm(train_loader):
            train_step(model,batch,train_loss_recoder,criterion,train=True)

        train_loss=train_loss_recoder.acc()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}")
        writer.add_scalar('Training/Loss', train_loss, epoch)

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in tqdm(val_loader):
                    train_step(model,batch,val_loss_recoder,criterion,train=False)

        val_loss=val_loss_recoder.acc()

        print(f"Epoch {epoch + 1}/{epochs},Validation Loss: {val_loss}")
        writer.add_scalar('Validation/Loss', val_loss, epoch)

        # Save model every 50 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{model_dir}/epoch_{epoch + 1}.pth')
            print(f"Model saved at epoch {epoch + 1}")

    print("Training complete.")
    writer.close()