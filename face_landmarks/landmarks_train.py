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
from utils.dataloader import AFLW2000Data
from face_landmarks.facemesh import FaceMeshBlock, FaceMesh


#####################################
# Training
#####################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Parameteres
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, '../data/AFLW2000')

#learning rate
lr = 10e-6

#number of training epochs
epoch_n = 200

#input image-mask size
image_size = 192
#root directory of project
root_dir = os.getcwd()

#training batch size
batch_size = 4

model = FaceMesh().to(device)

# Loss
class WingLoss(nn.Module):
    def __init__(self, width=5, curvature=0.5):
        super(WingLoss, self).__init__()
        self.width = width
        self.curvature = curvature
        self.C = self.width - self.width * np.log(1 + self.width / self.curvature)

    def forward(self, prediction, target):
        diff = target - prediction
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        idx_smaller = diff_abs < self.width
        idx_bigger = diff_abs >= self.width

        loss[idx_smaller] = self.width * torch.log(1 + diff_abs[idx_smaller] / self.curvature)
        loss[idx_bigger]  = loss[idx_bigger] - self.C
        loss = loss.mean()
        return loss

    
# training configurations
trainset = AFLW2000Data(data_dir = data_dir, size = image_size, train = True, train_test_split = 0.8)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)

testset = AFLW2000Data(data_dir = data_dir, size = image_size, train = False, train_test_split = 0.8)
testloader = DataLoader(testset, batch_size = batch_size, shuffle=True)

criterion = WingLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)


#use checkpoint model for training
load = False

if load:
    print('loading model')
    model.load_state_dict(torch.load('Face-real-2000.pt'))
    
train_loss = []
valid_loss = []

start_time = datetime.now()

for e in range(epoch_n):
    epoch_loss = 0
    
    print("######## Train ########")
    model.train()
    
    for i, data in enumerate(trainloader):
        image, label = data
        #print("==========DEBUG DATALOADER============")
        #plt.imshow(image[0,:,:,:].permute(1,2,0))
        #plt.scatter(label[0,0,:], label[0,1,:])
        #plt.savefig("dataloader_inspect.png")
        #break
        image = image.float().to(device)  # (b x 3 x 192 x 192)
        label = label.float().to(device)  # (b x 2 x 68)
        

        output, confidence = model(image) # output: (b, 204), confidence: (b, 1)
        loss = criterion(output.view(batch_size, 3, -1)[:, :2, :], label[:, :2, :])
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    print('Epoch %d / %d --- Loss: %.4f' % (e + 1, epoch_n, epoch_loss / trainset.__len__()))
    print(datetime.now())
    train_loss.append(epoch_loss / trainset.__len__())

    torch.save(model.state_dict(), 'Face-real-2000-wing.pt')

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
