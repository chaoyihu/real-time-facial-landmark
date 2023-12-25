import os
import cv2
import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader


class AFLW2000Data(Dataset):
    def __init__(self, data_dir, size, train = 'True', train_test_split = 0.9):

        super(AFLW2000Data, self).__init__()

        self.size = size
        self.inputs = []
        self.targets = []

        for fname in os.listdir(data_dir):
            if fname.endswith(".jpg"):
                self.inputs.append(fname)
            if fname.endswith(".mat"):
                self.targets.append(fname)

        self.inputs.sort()
        self.targets.sort()
        self.inputs = [os.path.join(data_dir, fname) for fname in self.inputs]
        self.targets = [os.path.join(data_dir, fname) for fname in self.targets]
        
        
    def __getitem__(self, idx):
        
        #load image and mask from index idx of your data
        input_ID = self.inputs[idx]
        target_ID = self.targets[idx]

        img = cv2.imread(input_ID)
        ldmks = np.array(loadmat(target_ID)['pt3d_68']) # (3, 68)
        original_x, original_y, channel = img.shape
        ratio_x = self.size / original_x
        ratio_y = self.size / original_y
        img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ldmks[0, :] *= ratio_x
        ldmks[1, :] *= ratio_y

        #return image and mask in tensors        
        image = torch.from_numpy(img)
        image = image.permute(2, 0, 1) # (192 * 192 *3) -> (3 * 192 * 192)
        ldmks = ldmks.reshape((3, 68))
        ldmks = torch.from_numpy(ldmks[:2, :])

        return image, ldmks
    
    def __len__(self):
        return len(self.inputs)
