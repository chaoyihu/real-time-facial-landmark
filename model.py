###############################
# Libraries | Utils | Settings
###############################
import os
import torch
from dataloader import AFLW2000Data

# from google.colab import drive
# drive.mount('/content/drive')

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


###############################
# Define the model
###############################
# face detector
from blazeface import BlazeFace

# face landmarks
from facemesh import FaceMesh


###############################
# Training
###############################
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

print("==== MODEL ====")
print(model)

face_detector = BlazeFace()
