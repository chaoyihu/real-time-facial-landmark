import os
import cv2
import numpy as np
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader


class LoadMixedData(Dataset):
    def __init__(self, size, train = 'True', train_test_split = 0.9):

        super(LoadMixedData, self).__init__()

        self.size = size
        self.inputs = []
        self.targets = set()

        datasets = ["AFLW2000", "afw", "helen_testset", "helen_trainset", 
                "ibug", "lfpw_1", "lfpw_2", "synthetic1000"]

        for dataset in datasets:
            data_dir = "../data/" + dataset
            for fname in os.listdir(data_dir):
                if fname.endswith("seg.png"):
                    continue
                if fname.endswith("jpg") or fname.endswith("png"):
                    self.inputs.append(os.path.join(data_dir, fname))
                if fname.endswith("mat") or fname.endswith("ldmks.txt") or fname.endswith("pts"):
                    self.targets.add(os.path.join(data_dir, fname))

        self.inputs.sort()
        
    def __getitem__(self, idx):
        '''
        load image and mask from index idx of your data
        '''
        input_dir = self.inputs[idx]

        # load original image
        img = cv2.imread(input_dir)
        original_h, original_w, channel = img.shape
        # load original labels
        ls = input_dir.split("/")
        fname = ls.pop().split(".")[0]
        ls.append(fname)
        target_dir = os.path.join(*ls)
        if target_dir.endswith("mirror"):
            target_dir = target_dir[:-7] + ".pts"
            with open(target_dir) as f:
                rows = [rows.strip() for rows in f]
            head = rows.index('{') + 1
            tail = rows.index('}')
            raw_points = rows[head:tail]
            ldmks = [[],[]]
            for point in raw_points:
                x, y = list(map(float, point.split()))
                ldmks[0].append(original_w - x)
                ldmks[1].append(y)
            ldmks = np.array(ldmks)
        elif target_dir + ".mat" in self.targets:
            target_dir = target_dir + ".mat"
            ldmks = np.array(loadmat(target_dir)['pt3d_68']) # (3, 68)
            ldmks = ldmks[:2, :]
        elif target_dir + "_ldmks.txt" in self.targets:
            target_dir = target_dir + "_ldmks.txt"
            ldmks = [[],[]]
            with open(target_dir, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                x, y = list(map(float, line.split(" ")))
                ldmks[0].append(x)
                ldmks[1].append(y)
            ldmks[0] = ldmks[0][:68]
            ldmks[1] = ldmks[1][:68]
            ldmks = np.array(ldmks)
        else:
            target_dir = target_dir + ".pts"
            with open(target_dir) as f:
                rows = [rows.strip() for rows in f]
            head = rows.index('{') + 1
            tail = rows.index('}')
            raw_points = rows[head:tail]
            ldmks = [[],[]]
            for point in raw_points:
                x, y = list(map(float, point.split()))
                ldmks[0].append(x)
                ldmks[1].append(y)
            ldmks = np.array(ldmks)

        # crop image and adjust labels to match
        center_x, center_y = np.mean(ldmks[0, :]), np.mean(ldmks[1, :])
        span_x, span_y = np.max(ldmks[0, :]) - np.min(ldmks[0, :]), np.max(ldmks[1, :]) - np.min(ldmks[1, :])
        xmin, xmax = max(0, center_x - span_x), min(original_w, center_x + span_x)
        ymin, ymax = max(0, center_y - span_y), min(original_h, center_y + span_y)
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        crop_h, crop_w, channels = img.shape
        ldmks[0, :] -= xmin
        ldmks[1, :] -= ymin

        # resize image and adjust labels to match
        ratio_x = self.size / crop_w
        ratio_y = self.size / crop_h
        img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ldmks[0, :] *= ratio_x
        ldmks[1, :] *= ratio_y

        #return image and mask in tensors 
        image = torch.from_numpy(img)
        image = image.permute(2, 0, 1) # (192 * 192 *3) -> (3 * 192 * 192)
        ldmks = torch.from_numpy(ldmks)

        return image, ldmks


    def __len__(self):
        return len(self.inputs)
