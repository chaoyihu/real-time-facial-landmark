################################
# Libraries | Utils | Settings
################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.optim as optim

from PIL import Image, ImageOps

import cv2
from datetime import datetime
import random

from scipy.io import loadmat

sys.path.append(os.path.join(sys.path[0], ".."))
from utils.dataloader import LoadMixedData 
from face_landmarks.facemesh import FaceMeshBlock, FaceMesh

# from google.colab import drive
# drive.mount('/content/drive')

#####################################
# Training
#####################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameters
root_dir = os.getcwd()   #root directory of project
lr = 10e-5   # leanrning rate
epoch_n = 200   #number of training epochs
image_size = 192   #input image-mask size
batch_size = 8    #training batch size

model = FaceMesh().to(device)

# Loss
# class WingLoss(nn.Module):
#     def __init__(self, width=5, curvature=0.5):
#         super(WingLoss, self).__init__()
#         self.width = width
#         self.curvature = curvature
#         self.C = self.width - self.width * np.log(1 + self.width / self.curvature)
# 
#     def forward(self, prediction, target):
#         diff = target - prediction
#         diff_abs = diff.abs()
#         loss = diff_abs.clone()
# 
#         idx_smaller = diff_abs < self.width
#         idx_bigger = diff_abs >= self.width
# 
#         loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
#         loss[idx_bigger]  = loss[idx_bigger] - self.C
#         loss = loss.mean()
#         return loss
# 
    
# training configurations
trainset = LoadMixedData(size = image_size, train = True, train_test_split = 0.8)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, drop_last=True)

testset = LoadMixedData(size = image_size, train = False, train_test_split = 0.8)
testloader = DataLoader(testset, batch_size = batch_size, shuffle=True, drop_last=True)

# criterion = WingLoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)


train_loss = []
valid_loss = []

#use checkpoint model for training
load = False
if load:
    print('loading model')
    cpt = torch.load('model_checkpoint.pth')
    model.load_state_dict(cpt['model_state_dict'])
    #optimizer.load_state_dict(cpt['optimizer_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00002)


start_time = datetime.now()

for e in range(epoch_n):
    
    print("######## Train ########")
    model.train()
    
    epoch_loss = 0
    for data in trainloader:
        image, label = data
        print("==========DEBUG DATALOADER============")
        plt.figure()
        plt.imshow(image[0,:,:,:].permute(1,2,0))
        plt.scatter(label[0,0,:], label[0,1,:])
        plt.savefig("dataloader_inspect.png")
        plt.close()
        image = image.float().to(device)  # (b x 3 x 192 x 192)
        label = label.float().to(device)  # (b x 2 x 68)

        output, confidence = model(image) # output: (b, 204), confidence: (b, 1)
        loss = criterion(output.view(batch_size, 3, -1)[:, :2, :], label[:, :2, :])
        loss.backward()
        epoch_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
    
    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
    print(datetime.now())
    train_loss.append(loss.item())

    print("######## Validation ########")
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            image, label = data

            image = image.float().to(device)
            label = label.float().to(device)

            pred, confidence = model(image)
            loss = criterion(pred.view(batch_size, 3, -1)[:, :2, :], label[:, :2, :])
            total_loss += loss.item()

            _, pred_labels = torch.max(pred, dim = 1)
            
        print('Loss: %.4f' % (total_loss / testset.__len__()))
        valid_loss.append(total_loss / testset.__len__())

    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        }, "model_checkpoint.pth")
    
end_time = datetime.now()


end_time = datetime.now()
delta = end_time - start_time
s = delta.total_seconds()
m, s = divmod(s, 60)
h, m = divmod(m, 60)
print(f"Start: {datetime.strftime(start_time, '%H:%M:%S')}")
print(f"End: {datetime.strftime(end_time, '%H:%M:%S')}")
print(f"Training time: {int(h)} h {int(m)} min {int(s)} s")


fig = plt.figure(figsize=(6,2))
ep = list(range(len(train_loss)))
ax1 = plt.subplot(111)
plt.plot(ep, train_loss, 'r', label="Train loss")
plt.plot(ep, valid_loss, 'b', label="Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("Loss.png")
