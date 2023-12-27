import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import glob
import os
import pickle
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F
import json
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
import datetime
import random
import itertools
import datetime
import torch.nn as nn
from pytorch_lightning import Trainer
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import scipy.spatial.transform as spt
import os
import pickle
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import gc  # Garbage collection
import open3d as o3d
from scipy.spatial import cKDTree
import math
import argparse
from datetime import datetime
#from memory_profiler import profile
import pdb
from pytorch_lightning.loggers import WandbLogger


class BehaveDataset(Dataset):
    def __init__(self, labels, cam_ids, frames_subclip, selected_keys, wandb, device):
        self.labels = labels
        self.cam_ids = cam_ids
        self.frames_subclip = frames_subclip
        self.selected_keys = selected_keys
        self.device = device
        self.data_info = []  # Store file path, camera id, subclip index range, and window indices

        base_path = '/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy'
        # base_path = '/scratch_net/biwidl307/lgermano/H2O/30fps_int_1frame_numpy'
        print(f"Initializing BehaveDataset with {len(labels)} labels and {len(cam_ids)} camera IDs.")

        for label in self.labels:
            for cam_id in self.cam_ids:
                file_path = os.path.join(base_path, label + '.pkl')
                print(file_path)
                if os.path.exists(file_path):
                    print(f"Found file: {file_path}", flush=True)
                    with open(file_path, 'rb') as f:

                        if len(self.labels) == 1:

                            self.dataset = pickle.load(f)
                            for start_idx in range(0, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                            #for start_idx in range(len(self.dataset[cam_id]) - 2 * self.frames_subclip, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                                end_idx = start_idx + self.frames_subclip
                                if end_idx <= len(self.dataset[cam_id]):
                                    self.data_info.append((file_path, cam_id, start_idx, end_idx))

                        else:

                            dataset = pickle.load(f)
                            for start_idx in range(0, len(dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                            #for start_idx in range(0, len(dataset[cam_id]) - self.frames_subclip):
                            #for start_idx in range(len(self.dataset[cam_id]) - 2 * self.frames_subclip, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                                end_idx = start_idx + self.frames_subclip
                                if end_idx <= len(dataset[cam_id]):
                                    self.data_info.append((file_path, cam_id, start_idx, end_idx))

    def __len__(self):
        print(len(self.data_info))
        return len(self.data_info)

    def __getitem__(self, idx):
        file_path, cam_id, start_idx, end_idx = self.data_info[idx]

        # Only possible if there is one training label
        if len(self.labels) == 1:
            subclip_data = self.dataset[cam_id][start_idx:end_idx]
            scene = self.dataset[cam_id][0]['scene']
        else:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)

            subclip_data = dataset[cam_id][start_idx:end_idx]
            scene = dataset[cam_id][0]['scene']

        items = [torch.tensor(np.vstack([subclip_data[i][key] for i in range(len(subclip_data))]), dtype=torch.float32) for key in self.selected_keys]

        return items, scene

class BehaveDatasetOffset(Dataset):
    def __init__(self, labels, cam_ids, frames_subclip, selected_keys, wandb, device):
        self.labels = labels
        self.cam_ids = cam_ids
        self.frames_subclip = frames_subclip
        self.selected_keys = selected_keys
        self.device = device
        self.data_info = []  # Store file path, camera id, subclip index range, and window indices

        base_path = '/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy'
        # base_path = '/scratch_net/biwidl307/lgermano/H2O/30fps_int_1frame_numpy'
        print(f"Initializing BehaveDataset with {len(labels)} labels and {len(cam_ids)} camera IDs.")

        for label in self.labels:
            for cam_id in self.cam_ids:
                file_path = os.path.join(base_path, label + '.pkl')
                print(file_path)
                if os.path.exists(file_path):
                    print(f"Found file: {file_path}", flush=True)
                    with open(file_path, 'rb') as f:

                        if len(self.labels) == 1:

                            self.dataset = pickle.load(f)
                            for start_idx in range(0, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                            #for start_idx in range(len(self.dataset[cam_id]) - 2 * self.frames_subclip, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                                end_idx = start_idx + self.frames_subclip
                                if end_idx <= len(self.dataset[cam_id]):
                                    self.data_info.append((file_path, cam_id, start_idx, end_idx))

                        else:

                            dataset = pickle.load(f)
                            for start_idx in range(0, len(dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                            #for start_idx in range(0, len(dataset[cam_id]) - self.frames_subclip):
                            #for start_idx in range(len(self.dataset[cam_id]) - 2 * self.frames_subclip, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                                end_idx = start_idx + self.frames_subclip
                                if end_idx <= len(dataset[cam_id]):
                                    self.data_info.append((file_path, cam_id, start_idx, end_idx))

    def __len__(self):
        print(len(self.data_info))
        return len(self.data_info)

    def __getitem__(self, idx):
        file_path, cam_id, start_idx, end_idx = self.data_info[idx]

        # Only possible if there is one training label
        if len(self.labels) == 1:
            subclip_data = self.dataset[cam_id][start_idx:end_idx]
            scene = self.dataset[cam_id][0]['scene']
        else:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)

            subclip_data = dataset[cam_id][start_idx:end_idx]
            scene = dataset[cam_id][0]['scene']
        
        # Initialize with zeros

        # For human and object parameters. The first is zero.
        
        items = []
        for key in self.selected_keys:
            tensors = []
            for i in range(len(subclip_data)):
                if i == 0:
                    # Use a tensor of zeros for the first element
                    zeros_tensor = torch.zeros_like(torch.tensor(subclip_data[i][key], dtype=torch.float32))
                    tensors.append(zeros_tensor)
                else:
                    # Compute the difference with the previous value
                    diff = torch.tensor(subclip_data[i][key], dtype=torch.float32) - torch.tensor(subclip_data[i-1][key], dtype=torch.float32)
                    tensors.append(diff)
            # Stack the tensors for each key
            stacked_tensors = torch.stack(tensors)
            items.append(stacked_tensors)

        return items, scene

class BehaveDatasetOffset2(Dataset):
    def __init__(self, labels, cam_ids, frames_subclip, selected_keys, wandb, device):
        self.labels = labels
        self.cam_ids = cam_ids
        self.frames_subclip = frames_subclip
        self.selected_keys = selected_keys
        self.device = device
        self.data_info = []  # Store file path, camera id, subclip index range, and window indices

        base_path = '/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy'
        # base_path = '/scratch_net/biwidl307/lgermano/H2O/30fps_int_1frame_numpy'
        print(f"Initializing BehaveDataset with {len(labels)} labels and {len(cam_ids)} camera IDs.")

        for label in self.labels:
            for cam_id in self.cam_ids:
                file_path = os.path.join(base_path, label + '.pkl')
                print(file_path)
                if os.path.exists(file_path):
                    print(f"Found file: {file_path}", flush=True)
                    with open(file_path, 'rb') as f:

                        if len(self.labels) == 1:

                            self.dataset = pickle.load(f)
                            for start_idx in range(0, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                            #for start_idx in range(len(self.dataset[cam_id]) - 2 * self.frames_subclip, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                                end_idx = start_idx + self.frames_subclip
                                if end_idx <= len(self.dataset[cam_id]):
                                    self.data_info.append((file_path, cam_id, start_idx, end_idx))

                        else:

                            dataset = pickle.load(f)
                            for start_idx in range(0, len(dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                            #for start_idx in range(0, len(dataset[cam_id]) - self.frames_subclip):
                            #for start_idx in range(len(self.dataset[cam_id]) - 2 * self.frames_subclip, len(self.dataset[cam_id]) - self.frames_subclip, self.frames_subclip):
                                end_idx = start_idx + self.frames_subclip
                                if end_idx <= len(dataset[cam_id]):
                                    self.data_info.append((file_path, cam_id, start_idx, end_idx))

    def __len__(self):
        print(len(self.data_info))
        return len(self.data_info)

    def __getitem__(self, idx):
        file_path, cam_id, start_idx, end_idx = self.data_info[idx]

        # Only possible if there is one training label
        if len(self.labels) == 1:
            subclip_data = self.dataset[cam_id][start_idx:end_idx]
            scene = self.dataset[cam_id][0]['scene']
        else:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)

            subclip_data = dataset[cam_id][start_idx:end_idx]
            scene = dataset[cam_id][0]['scene']
        
        # Initialize with zeros
        # For object parameters use delta_object - delta_human. The first is zero.
        
        items = []
        for key in self.selected_keys:
            tensors = []
            for i in range(len(subclip_data)):
                if i == 0:
                    # Use a tensor of zeros for the first element
                    zeros_tensor = torch.zeros_like(torch.tensor(subclip_data[i][key], dtype=torch.float32))
                    tensors.append(zeros_tensor)
                else:
                    # Compute the difference with the previous value
                    diff = torch.tensor(subclip_data[i][key], dtype=torch.float32) - torch.tensor(subclip_data[i-1][key], dtype=torch.float32)
                    tensors.append(diff)
            # Stack the tensors for each key
            stacked_tensors = torch.stack(tensors)
            items.append(stacked_tensors)

        for i in range(len(subclip_data)):
            items[2][i] = torch.tensor(items[2][i], dtype=torch.float32) - torch.tensor(items[0][i][:3], dtype=torch.float32)
            items[3][i] = torch.tensor(items[3][i], dtype=torch.float32) - torch.tensor(items[1][i][0], dtype=torch.float32)                      

        return items, scene

class BehaveDataModule(pl.LightningDataModule):
    def __init__(self, dataset, split, batch_size):
        super(BehaveDataModule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.split = split

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []
        train_identifiers = []
        test_identifiers = []

        for idx, (data, scene_name) in enumerate(self.dataset):
            scene = scene_name  # Assuming the scene name is the second element in the tuple
            if scene in self.split['train']:
                self.train_indices.append(idx)
                train_identifiers.append(scene)
            elif scene in self.split['test']:
                self.test_indices.append(idx)
                test_identifiers.append(scene)

        self.val_indices = self.test_indices  # Assuming validation and training sets are the same

        # Uncomment to print identifiers in train and test sets
        print(f"Identifiers in train set: {set(train_identifiers)}", flush=True)
        print(f"Identifiers in test set: {set(test_identifiers)}", flush=True)

    def train_dataloader(self):
        train_dataset = Subset(self.dataset, self.train_indices)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=0)

    def val_dataloader(self):
        val_dataset = Subset(self.dataset, self.val_indices)
        return DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True, num_workers=0)

    def test_dataloader(self):
        test_dataset = Subset(self.dataset, self.test_indices)
        return DataLoader(test_dataset, batch_size=self.batch_size, drop_last=True, num_workers=0)
