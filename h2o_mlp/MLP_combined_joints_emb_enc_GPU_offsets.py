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

best_overall_avg_loss_val = float('inf')
best_params = None

# Set the WANDB_CACHE_DIR environment variable
os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"

#learning_rate_range = [1e-3]#, 1e-1]#, 1e-4, 1e-5]
epochs_range = [3000]
learning_rate_range = [1e-3]#, 5e-3, 1e-2, 5e-2]
batch_size_range = [16]#[32, 64, 128]
dropout_rate_range = [0]
alpha_range =[100]#[0.01,0.1,1,10,100]
lambda_1_range = [10]#[0.01,0.1,1,10,100]
lambda_2_range = [0.01]#[0.0001,0.001,0.01,0.1,1]
L_range = [4]#,6,10]
layer_sizes_range_1 = [
    #[128, 128, 64, 64, 32],
    # [256, 256, 128, 128, 64],
    # [512, 512, 256, 256, 128],
    # [1024, 1024, 512, 512, 256],
    # [256, 256, 256, 128, 64, 32],
    # [512, 512, 512, 256, 128, 64],
    # [1024, 1024, 1024, 512, 256, 128],
    # [128, 128, 128, 64, 64, 32, 32],
    [256, 256, 256],
    #[256, 256, 256, 256, 256, 256]  
]
layer_sizes_range_3 = [
    #[128, 128, 64, 64, 32],
    # [256, 256, 128, 128, 64],
    # [512, 512, 256, 256, 128],
    # [1024, 1024, 512, 512, 256],
    # [256, 256, 256, 128, 64, 32],
    # [512, 512, 512, 256, 128, 64],
    # [1024, 1024, 1024, 512, 256, 128],
    # [128, 128, 128, 64, 64, 32, 32],
    [64, 128, 256, 128, 64]
    #[256, 256, 256],
]
optimizer_list = ["SGD", "Adagrad", "RMSprop", "AdamW", "Adadelta", "LBFGS"]


for lr, bs, dr, layers_1, layers_3, alpha, lambda_1, lambda_2, l, epochs, optimizer_name in itertools.product(
    learning_rate_range, batch_size_range, dropout_rate_range, layer_sizes_range_1, layer_sizes_range_3, alpha_range, lambda_1_range, lambda_2_range, L_range, epochs_range, optimizer_list
):

    LEARNING_RATE = lr
    BATCH_SIZE = bs
    DROPOUT_RATE = dr
    LAYER_SIZES_1 = layers_1
    LAYER_SIZES_3 = layers_3
    # Initialize the input to 24 joints
    INITIAL_OBJ_PRED = torch.rand((BATCH_SIZE, 24))
    ALPHA = alpha
    LAMBDA_1 = lambda_1
    LAMBDA_2 = lambda_2
    EPOCHS = epochs
    L = l
    OPTIMIZER = optimizer_name

    #trainer = Trainer(log_every_n_steps=BATCH_SIZE)  # Log every n steps

    wandb.init(
        project="MLP",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "MLP",
            "dataset": "BEHAVE",
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "layer_sizes_1": LAYER_SIZES_1,
            "layer_sizes_3": LAYER_SIZES_3,
            "alpha": ALPHA,
            "lambda_1": LAMBDA_1,
            "lambda_2": LAMBDA_2,
            "L": L,
            "epochs": EPOCHS,
            "optimizer": OPTIMIZER
        },
    )


    def load_pickle(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def load_config(camera_id, base_path, Date='Date07'):
        config_path = os.path.join(
            base_path, "calibs", Date, "config", str(camera_id), "config.json"
        )
        with open(config_path, "r") as f:
            return json.load(f)

    def load_split_from_path(path):
        with open(path, "r") as file:
            split_dict = json.load(file)
        return split_dict
    
    def linear_interpolate(value1, value2, i):
        return value1 + (i/3) * (value2 - value1)
    
    def slerp(p0, p1, t):
        # Convert axis-angle to quaternion
        q0 = spt.Rotation.from_rotvec(p0).as_quat()
        q1 = spt.Rotation.from_rotvec(p1).as_quat()
        
        # Normalize quaternions
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        
        # SLERP
        cosine = np.dot(q0, q1)
        
        # Ensure the shortest path is taken
        if cosine < 0.0:
            q1 = -q1
            cosine = -cosine
            
        # If q0 and q1 are very close, use linear interpolation as an approximation
        if abs(cosine) >= 1.0 - 1e-10:
            return p0 + t * (p1 - p0)

        omega = np.arccos(cosine)
        so = np.sin(omega)
        res_quat = (np.sin((1.0-t)*omega) / so) * q0 + (np.sin(t*omega)/so) * q1
        
        # Convert quaternion back to axis-angle
        res_rotvec = spt.Rotation.from_quat(res_quat).as_rotvec()
        return res_rotvec
  
    def slerp_rotations(p0, p1, t):
        num_joints = len(p0) // 3
        interpolated_rotations = np.empty_like(p0)

        for i in range(num_joints):
            start_idx = i * 3
            end_idx = (i + 1) * 3
            
            joint_rot0 = p0[start_idx:end_idx]
            joint_rot1 = p1[start_idx:end_idx]
            
            interpolated_rot = slerp(joint_rot0, joint_rot1, t)
            interpolated_rotations[start_idx:end_idx] = interpolated_rot

        return interpolated_rotations

    def interpolate_frames(all_data_frames):
        interpolated_frames = []

        for idx in range(len(all_data_frames)-1):
            frame1 = all_data_frames[idx]
            frame2 = all_data_frames[idx+1]

            # Original frame
            interpolated_frames.append(frame1)

            # Interpolated frames

            for i in range(1, 10):
                interpolated_frame = frame1.copy()
                t = i / 10.0  # Assuming you want to interpolate at 1/3 and 2/3 positions between frame1 and frame2
                interpolated_frame['pose'] = slerp_rotations(frame1['pose'], frame2['pose'], t)
                interpolated_frame['trans'] = linear_interpolate(frame1['trans'], frame2['trans'], t)
                interpolated_frame['obj_pose'] = slerp_rotations(frame1['obj_pose'], frame2['obj_pose'], t)
                interpolated_frame['obj_trans'] = linear_interpolate(frame1['obj_trans'], frame2['obj_trans'], t)
                
                interpolated_frames.append(interpolated_frame)            

        # Adding the last original frame
        interpolated_frames.append(all_data_frames[-1])

        return interpolated_frames

    def transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params):
        # Convert axis-angle representation to rotation matrix
        R_w = Rotation.from_rotvec(pose[:3]).as_matrix()
        
        # Build transformation matrix of mesh in world coordinates
        T_mesh = np.eye(4)
        T_mesh[:3, :3] = R_w
        T_mesh[:3, 3] = trans
        
        # Extract rotation and translation of camera from world coordinates
        R_w_c = np.array(cam_params['rotation']).reshape(3, 3)
        t_w_c = np.array(cam_params['translation']).reshape(3,)
        
        # Build transformation matrix of camera in world coordinates
        T_cam = np.eye(4)
        T_cam[:3, :3] = R_w_c
        T_cam[:3, 3] = t_w_c

        T_cam = T_cam.astype(np.float64)
        T_mesh = T_mesh.astype(np.float64)
        T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh
        
        # Extract transformed pose and translation of mesh in camera coordinate frame
        transformed_pose = Rotation.from_matrix(T_mesh_in_cam[:3, :3]).as_rotvec().flatten()
        transformed_pose = np.concatenate([transformed_pose, pose[3:]]).flatten()
        transformed_trans = T_mesh_in_cam[:3, 3].flatten()
        return np.concatenate([transformed_pose, transformed_trans])

    def transform_object_to_camera_frame(data, camera1_params, cam_params):
        """Transform object's position and orientation to another camera frame using relative transformation."""
        # Convert the axis-angle rotation to a matrix

        R_w = Rotation.from_rotvec(data["angle"]).as_matrix()

        # Build transformation matrix of mesh in world coordinates
        T_mesh = np.eye(4)
        T_mesh[:3, :3] = R_w
        T_mesh[:3, 3] = data["trans"]
        
        # Extract rotation and translation of camera from world coordinates
        R_w_c = np.array(cam_params['rotation']).reshape(3, 3)
        t_w_c = np.array(cam_params['translation']).reshape(3,)
        
        # Build transformation matrix of camera in world coordinates
        T_cam = np.eye(4)
        T_cam[:3, :3] = R_w_c
        T_cam[:3, 3] = t_w_c

        T_cam = T_cam.astype(np.float64)
        T_mesh = T_mesh.astype(np.float64)
        T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh
        transformed_trans = T_mesh_in_cam[:3, 3].flatten()

        return transformed_trans
    
    def render_smpl(transformed_pose, transformed_trans, betas):
    
        #print("Start of render_smpl function.")
        
        batch_size = 1
        #print(f"batch_size: {batch_size}")

        # Create the SMPL layer
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='male',
            model_root='/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch/native/models/')
        #print("SMPL_Layer created.")

        # Process pose parameters
        pose_params_start = torch.tensor(transformed_pose[:3], dtype=torch.float32)
        pose_params_rest = torch.tensor(transformed_pose[3:72], dtype=torch.float32)
        pose_params_rest[-6:] = 0
        pose_params = torch.cat([pose_params_start, pose_params_rest]).unsqueeze(0).repeat(batch_size, 1)
        #print(f"pose_params shape: {pose_params.shape}")

        shape_params = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        #print(f"shape_params shape: {shape_params.shape}")

        obj_trans = torch.tensor(transformed_trans, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        #print(f"obj_trans shape: {obj_trans.shape}")

        # GPU mode
        cuda = torch.cuda.is_available()
        #print(f"CUDA available: {cuda}")
        device = torch.device("cuda:0" if cuda else "cpu")
        #print(f"Device: {device}")
        
        pose_params = pose_params.to(device)
        shape_params = shape_params.to(device)
        obj_trans = obj_trans.to(device)
        smpl_layer = smpl_layer.to(device)
        #print("All tensors and models moved to device.")

        # Forward from the SMPL layer
        verts, J = smpl_layer(pose_params, th_betas=shape_params, th_trans=obj_trans)

        J = J.squeeze(0)
        # Extracting joints from SMPL skeleton
        pelvis = J[0]
        left_hip = J[1]
        right_hip = J[2]
        spine1 = J[3]
        left_knee = J[4]
        right_knee = J[5]
        spine2 = J[6]
        left_ankle = J[7]
        right_ankle = J[8]
        spine3 = J[9]
        left_foot = J[10]
        right_foot = J[11]
        neck = J[12]
        left_collar = J[13]
        right_collar = J[14]
        head = J[15]
        left_shoulder = J[16]
        right_shoulder = J[17]
        left_elbow = J[18]
        right_elbow = J[19]
        left_wrist = J[20]
        right_wrist = J[21]
        left_hand = J[22]
        right_hand = J[23]

        # Creating a list with all joints
        selected_joints = [pelvis, left_hip, right_hip, spine1, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, 
                        left_foot, right_foot, neck, left_collar, right_collar, head, left_shoulder, right_shoulder, 
                        left_elbow, right_elbow, left_wrist, right_wrist, left_hand, right_hand]
        
        # selected_joints = [pelvis, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3, 
        #                  left_foot, right_foot, head, left_shoulder, right_shoulder, left_hand, right_hand]      
        return selected_joints

    def load_frames_distance_regressor(interpolated_data_frames):
        
        reprojected_smpl_cam1_list = []
        reprojected_smpl_cam0_list = []
        reprojected_smpl_cam2_list = []
        reprojected_smpl_cam3_list = []

        reprojected_obj_cam1_list = []
        reprojected_obj_cam0_list = []
        reprojected_obj_cam2_list = []
        reprojected_obj_cam3_list = []
        
        identifiers = []

        # prev_smpl_data = None                
        # Date = 'Date07'
        base_path = "/scratch_net/biwidl307_second/lgermano/behave"

        # Process interpolated frames
        for idx, frame_data in enumerate(interpolated_data_frames):

            pose = frame_data['pose'][:72]
            trans = frame_data['trans']
            betas = frame_data['betas']
            obj_pose = frame_data['obj_pose']
            obj_trans = frame_data['obj_trans']
            #scene_name = frame_data['scene'] 
                            
            for cam_id in [1, 0, 2, 3]:
                #print(f"\nProcessing for camera {cam_id}...")
                
                camera1_params = load_config(1, base_path, 'Date07')
                cam_params = load_config(cam_id, base_path, 'Date07')
                transformed_smpl = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params)
                        
                if cam_id == 1:
                    
                    reprojected_smpl_cam1_list.append(transformed_smpl)

                    # Produce labels: distance joint-obj_trans
                    selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]
                    distances = np.asarray([np.linalg.norm(obj_trans - joint) for joint in selected_joints])

                    reprojected_obj_cam1_list.append(distances)

                    #print(f"Distances in cam {cam_id}: {distances}.")

                    ########################################
                    # # POSE + OBJ TRANS --> DISTANCES
                    # data = {}
                    # data['angle'] = obj_pose
                    # data['trans'] = obj_trans                 
                    # transformed_smpl[-3:] =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    # reprojected_smpl_cam1_list.append(transformed_smpl)

                    #########################################
                if cam_id == 0:
                    # data = {}
                    # data['angle'] = obj_pose
                    # data['trans'] = obj_trans                 
                    # transformed_smpl[-3:] =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    reprojected_smpl_cam0_list.append(transformed_smpl)
                    reprojected_obj_cam0_list.append(distances)
                if cam_id == 2:

                    # data = {}
                    # data['angle'] = obj_pose
                    # data['trans'] = obj_trans                 
                    # transformed_smpl[-3:] =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    reprojected_smpl_cam2_list.append(transformed_smpl)
                    reprojected_obj_cam2_list.append(distances)
                if cam_id == 3:
                    # data = {}
                    # data['angle'] = obj_pose
                    # data['trans'] = obj_trans                 
                    # transformed_smpl[-3:] =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    reprojected_smpl_cam3_list.append(transformed_smpl)
                    reprojected_obj_cam3_list.append(distances)
            
            #identifier = filename.split('/')[6]
            identifier = "Date07_Sub04_yogaball_play"
            identifiers.append(identifier)

        return reprojected_smpl_cam1_list, reprojected_smpl_cam0_list, reprojected_smpl_cam2_list, reprojected_smpl_cam3_list, reprojected_obj_cam1_list, \
            reprojected_obj_cam0_list, reprojected_obj_cam2_list, reprojected_obj_cam3_list, identifiers

    def load_frames_trilateration(interpolated_data_frames):
        
        reprojected_smpl_cam1_list = []
        reprojected_smpl_cam0_list = []
        reprojected_smpl_cam2_list = []
        reprojected_smpl_cam3_list = []

        reprojected_obj_cam1_list = []
        reprojected_obj_cam0_list = []
        reprojected_obj_cam2_list = []
        reprojected_obj_cam3_list = []
        
        identifiers = []

        # prev_smpl_data = None                
        # Date = 'Date07'
        # base_path = "/scratch_net/biwidl307_second/lgermano/behave"

        # # distance regressor - USING only to occupy first 24 places
        # model_dist_path = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_offset_trans_smpl_augmented_resilient-wind-1110.pt"
        # model_dist = torch.load(model_dist_path)
        # # Set the model to evaluation mode
        # model_dist.eval()

        # Process interpolated frames
        for idx, frame_data in enumerate(interpolated_data_frames):

            pose = frame_data['pose'][:72]
            trans = frame_data['trans']
            betas = frame_data['betas']
            obj_pose = frame_data['obj_pose']
            obj_trans = frame_data['obj_trans']
            #scene_name = frame_data['scene'] 
                            
            for cam_id in [1, 0, 2, 3]:
                #print(f"\nProcessing for camera {cam_id}...")
                
                camera1_params = load_config(1, base_path, 'Date07')
                cam_params = load_config(cam_id, base_path, 'Date07')
                transformed_smpl = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params)
                        
                if cam_id == 1:
                    
                    reprojected_obj_cam1_list.append(obj_trans)

                    selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                    # input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                    # input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                    # # Use the neural network model to predict the offset in the object's pose
                    # candidate_distances = model_dist(input_tensor)
                    # ##print(candidate_distances)
                    # distances = candidate_distances.detach().numpy().flatten() 
                    distances = np.zeros(24)
                    ##print(selected_joints)
                    transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                    ##print(transformed_smpl)
                    reprojected_smpl_cam1_list.append(transformed_smpl)

                    #print(f"Distances in cam {cam_id}: {distances}.")
                    
                if cam_id == 0:
                    data = {}
                    data['angle'] = obj_pose
                    data['trans'] = obj_trans                 
                    obj_trans =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    reprojected_obj_cam0_list.append(obj_trans)

                    selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                    # input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                    # input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                    # # Use the neural network model to predict the offset in the object's pose
                    # candidate_distances = model_dist(input_tensor)
                    # ##print(candidate_distances)
                    # distances = candidate_distances.detach().numpy().flatten()
                    distances = np.zeros(24)
                    
                    transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                    reprojected_smpl_cam0_list.append(transformed_smpl)
                
                if cam_id == 2:

                    data = {}
                    data['angle'] = obj_pose
                    data['trans'] = obj_trans                 
                    obj_trans =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    reprojected_obj_cam2_list.append(obj_trans)

                    selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                    # input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                    # input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                    # # Use the neural network model to predict the offset in the object's pose
                    # candidate_distances = model_dist(input_tensor)
                    # ##print(candidate_distances)
                    # distances = candidate_distances.detach().numpy().flatten() 
                    distances = np.zeros(24)
                    
                    transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                    reprojected_smpl_cam2_list.append(transformed_smpl)
                if cam_id == 3:
                    data = {}
                    data['angle'] = obj_pose
                    data['trans'] = obj_trans                 
                    obj_trans =  transform_object_to_camera_frame(data, camera1_params, cam_params)

                    reprojected_obj_cam3_list.append(obj_trans)

                    selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                    # input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                    # input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                    # # Use the neural network model to predict the offset in the object's pose
                    # candidate_distances = model_dist(input_tensor)
                    # ##print(candidate_distances)
                    # distances = candidate_distances.detach().numpy().flatten()
                    distances = np.zeros(24)
                    
                    transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                    reprojected_smpl_cam3_list.append(transformed_smpl)
                
            #identifier = filename.split('/')[6]
            identifier = "Date07_Sub04_yogaball_play"
            identifiers.append(identifier)

        return reprojected_smpl_cam1_list, reprojected_smpl_cam0_list, reprojected_smpl_cam2_list, reprojected_smpl_cam3_list, reprojected_obj_cam1_list, \
                reprojected_obj_cam0_list, reprojected_obj_cam2_list, reprojected_obj_cam3_list, identifiers
    
    class CustomCyclicLR(_LRScheduler):
        def __init__(self, optimizer, base_lr=5e-9, max_lr=5e-5, step_size=2000, mode='triangular', gamma=1.0, last_epoch=-1):
            self.base_lr = base_lr
            self.max_lr = max_lr
            self.step_size = step_size
            self.mode = mode
            self.gamma = gamma
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            new_lr = []
            for base_lr in self.base_lrs:
                cycle = math.floor(1 + self.last_epoch / (2 * self.step_size))
                x = abs(self.last_epoch / self.step_size - 2 * cycle + 1)
                if self.mode == 'triangular':
                    lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
                elif self.mode == 'triangular2':
                    lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / float(2 ** (cycle - 1))
                elif self.mode == 'exp_range':
                    lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (self.gamma**(self.last_epoch))
                else:
                    raise ValueError(f"Invalid mode: {self.mode}")
                new_lr.append(lr)
            return new_lr

    class MLP1(pl.LightningModule):

        def __init__(self, input_dim, middle_dim):
            super(MLP1, self).__init__()

            self.automatic_optimization = False

            # Use layer_sizes from wandb.config to create the architecture
            layer_sizes = [input_dim] + wandb.config.layer_sizes_1 + [middle_dim]
            self.linears = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
            
            # Dropout layer
            self.dropout = torch.nn.Dropout(wandb.config.dropout_rate)

            self.leaky_relu = nn.LeakyReLU(0.01) 

            # He initialization
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    init.kaiming_normal_(m.weight)
                    init.zeros_(m.bias)
            
            # Initialize validation_losses
            self.validation_losses = []

        def forward(self, x):
            for i, linear in enumerate(self.linears[:-1]):
                x = linear(x)
                x = F.relu(x)  # Activation function
                x = self.dropout(x)
                
            x = self.linears[-1](x)  # No activation for the last layer
            return x

        # def training_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
            
        #     # Compute the predictions
        #     y_hat = self(x)
        #     y_hat_cam0 = self(x_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     y_hat_cam3 = self(x_cam3)
            
        #     # # Compute the losses using geodesic distance
        #     # loss_original = geodesic_loss(y_hat, y)
        #     # loss_cam0 = geodesic_loss(y_hat_cam0, y_cam0)
        #     # loss_cam2 = geodesic_loss(y_hat_cam2, y_cam2)
        #     # loss_cam3 = geodesic_loss(y_hat_cam3, y_cam3)


        #     # Compute the losses using Mean Squared Error (MSE) - trans
        #     loss_original = F.mse_loss(y_hat, y)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
            
        #     # Average the losses
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
            
        #     # Log the average loss
        #     wandb.log({"loss_train": avg_loss.item()})#, step=self.current_epoch)       
        #     self.manual_backward(avg_loss)
        #     optimizer = self.optimizers()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     return avg_loss

        # def validation_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
        #     y_hat = self(x)
        #     loss_original = F.mse_loss(y_hat, y)
        #     y_hat_cam0 = self(x_cam0)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     y_hat_cam3 = self(x_cam3)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
        #     self.validation_losses.append(avg_loss)
        #     wandb.log({"loss_val": avg_loss.item()})#, step=self.current_epoch)
        #     return {'val_loss': avg_loss}

        # def on_validation_epoch_end(self):
        #     avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
        #     self.log('loss_val', avg_val_loss, prog_bar=True, logger=True)
        #     wandb.log({"avg_loss_val": avg_val_loss.item()})#, step=self.current_epoch)
        #     self.log_scheduler_info(avg_val_loss.item())
        #     self.validation_losses = []  # reset for the next epoch

        # def log_scheduler_info(self, val_loss):
        #     scheduler = self.lr_schedulers()
        #     if isinstance(scheduler, list):
        #         scheduler = scheduler[0]
            
        #     # Log learning rate of the optimizer
        #     for idx, param_group in enumerate(self.optimizers().param_groups):
        #         wandb.log({f"learning_rate_{idx}": param_group['lr']})
            
        #     # Log best metric value seen so far by the scheduler
        #     best_metric_val = scheduler.best
        #     wandb.log({"best_val_loss": best_metric_val})

        #     # Log number of epochs since last improvements
        #     epochs_since_improvement = scheduler.num_bad_epochs
        #     wandb.log({"epochs_since_improvement": epochs_since_improvement})
            
        #     # Manually step the scheduler
        #     scheduler.step(val_loss)
            
        # def test_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
        #     y_hat = self(x)
        #     loss_original = F.mse_loss(y_hat, y)
        #     y_hat_cam0 = self(x_cam0)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     y_hat_cam3 = self(x_cam3)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
        #     wandb.log({"loss_test": avg_loss.item()})
        #     return avg_loss

        # def configure_optimizers(self):
        #     optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        #     scheduler = {
        #         'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.5),
        #         'monitor': 'loss_val',
        #         'interval': 'epoch',
        #         'frequency': 1
        #     }
        #     return [optimizer], [scheduler]

    class MLP3(pl.LightningModule):

        def __init__(self, input_dim, middle_dim):
            super(MLP3, self).__init__()

            self.automatic_optimization = False

            # Use layer_sizes from wandb.config to create the architecture
            layer_sizes = [input_dim] + wandb.config.layer_sizes_3 + [middle_dim]
            self.linears = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
            
            # Dropout layer
            self.dropout = torch.nn.Dropout(wandb.config.dropout_rate)

            self.leaky_relu = nn.LeakyReLU(0.01) 

            # He initialization
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    init.kaiming_normal_(m.weight)
                    init.zeros_(m.bias)
            
            # Initialize validation_losses
            self.validation_losses = []

        def forward(self, x):
            for i, linear in enumerate(self.linears[:-1]):
                x = linear(x)
                x = F.relu(x)  # Activation function
                x = self.dropout(x)
                
            x = self.linears[-1](x)  # No activation for the last layer
            return x

        # def training_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
            
        #     # Compute the predictions
        #     y_hat = self(x)
        #     y_hat_cam0 = self(x_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     y_hat_cam3 = self(x_cam3)
            
        #     # # Compute the losses using geodesic distance
        #     # loss_original = geodesic_loss(y_hat, y)
        #     # loss_cam0 = geodesic_loss(y_hat_cam0, y_cam0)
        #     # loss_cam2 = geodesic_loss(y_hat_cam2, y_cam2)
        #     # loss_cam3 = geodesic_loss(y_hat_cam3, y_cam3)


        #     # Compute the losses using Mean Squared Error (MSE) - trans
        #     loss_original = F.mse_loss(y_hat, y)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
            
        #     # Average the losses
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
            
        #     # Log the average loss
        #     wandb.log({"loss_train": avg_loss.item()})#, step=self.current_epoch)       
        #     self.manual_backward(avg_loss)
        #     optimizer = self.optimizers()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     return avg_loss

        # def validation_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
        #     y_hat = self(x)
        #     loss_original = F.mse_loss(y_hat, y)
        #     y_hat_cam0 = self(x_cam0)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     y_hat_cam3 = self(x_cam3)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
        #     self.validation_losses.append(avg_loss)
        #     wandb.log({"loss_val": avg_loss.item()})#, step=self.current_epoch)
        #     return {'val_loss': avg_loss}

        # def on_validation_epoch_end(self):
        #     avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
        #     self.log('loss_val', avg_val_loss, prog_bar=True, logger=True)
        #     wandb.log({"avg_loss_val": avg_val_loss.item()})#, step=self.current_epoch)
        #     self.log_scheduler_info(avg_val_loss.item())
        #     self.validation_losses = []  # reset for the next epoch

        # def log_scheduler_info(self, val_loss):
        #     scheduler = self.lr_schedulers()
        #     if isinstance(scheduler, list):
        #         scheduler = scheduler[0]
            
        #     # Log learning rate of the optimizer
        #     for idx, param_group in enumerate(self.optimizers().param_groups):
        #         wandb.log({f"learning_rate_{idx}": param_group['lr']})
            
        #     # Log best metric value seen so far by the scheduler
        #     best_metric_val = scheduler.best
        #     wandb.log({"best_val_loss": best_metric_val})

        #     # Log number of epochs since last improvements
        #     epochs_since_improvement = scheduler.num_bad_epochs
        #     wandb.log({"epochs_since_improvement": epochs_since_improvement})
            
        #     # Manually step the scheduler
        #     scheduler.step(val_loss)
            
        # def test_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
        #     y_hat = self(x)
        #     loss_original = F.mse_loss(y_hat, y)
        #     y_hat_cam0 = self(x_cam0)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     y_hat_cam3 = self(x_cam3)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
        #     wandb.log({"loss_test": avg_loss.item()})
        #     return avg_loss

        # def configure_optimizers(self):
        #     optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        #     scheduler = {
        #         'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.5),
        #         'monitor': 'loss_val',
        #         'interval': 'epoch',
        #         'frequency': 1
        #     }
        #     return [optimizer], [scheduler]

    class MLP_trilateration(pl.LightningModule):

        def __init__(self, input_dim, output_dim):
            super(MLP_trilateration, self).__init__()

            self.automatic_optimization = False

            # Use layer_sizes from wandb.config to create the architecture
            layer_sizes = [input_dim] + wandb.config.layer_sizes_1 + [output_dim]
            self.linears = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
            
            # Dropout layer
            self.dropout = torch.nn.Dropout(wandb.config.dropout_rate)

            self.leaky_relu = nn.LeakyReLU(0.01) 

            # He initialization
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    init.kaiming_normal_(m.weight)
                    init.zeros_(m.bias)
            
            # Initialize validation_losses
            self.validation_losses = []

        def forward(self, x):
            for i, linear in enumerate(self.linears[:-1]):
                x = linear(x)
                x = F.relu(x)  # Activation function
                x = self.dropout(x)
                
            x = self.linears[-1](x)  # No activation for the last layer
            return x

        # def training_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
            
        #     # Compute the predictions
        #     y_hat = self(x)
        #     y_hat_cam0 = self(x_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     y_hat_cam3 = self(x_cam3)
            
        #     # # Compute the losses using geodesic distance
        #     # loss_original = geodesic_loss(y_hat, y)
        #     # loss_cam0 = geodesic_loss(y_hat_cam0, y_cam0)
        #     # loss_cam2 = geodesic_loss(y_hat_cam2, y_cam2)
        #     # loss_cam3 = geodesic_loss(y_hat_cam3, y_cam3)


        #     # Compute the losses using Mean Squared Error (MSE) - trans
        #     loss_original = F.mse_loss(y_hat, y)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
            
        #     # Average the losses
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
            
        #     # Log the average loss
        #     wandb.log({"loss_train": avg_loss.item()})#, step=self.current_epoch)       
        #     self.manual_backward(avg_loss)
        #     optimizer = self.optimizers()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     return avg_loss

        # def validation_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
        #     y_hat = self(x)
        #     loss_original = F.mse_loss(y_hat, y)
        #     y_hat_cam0 = self(x_cam0)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     y_hat_cam3 = self(x_cam3)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
        #     self.validation_losses.append(avg_loss)
        #     wandb.log({"loss_val": avg_loss.item()})#, step=self.current_epoch)
        #     return {'val_loss': avg_loss}

        # def on_validation_epoch_end(self):
        #     avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
        #     self.log('loss_val', avg_val_loss, prog_bar=True, logger=True)
        #     wandb.log({"avg_loss_val": avg_val_loss.item()})#, step=self.current_epoch)
        #     self.log_scheduler_info(avg_val_loss.item())
        #     self.validation_losses = []  # reset for the next epoch

        # def log_scheduler_info(self, val_loss):
        #     scheduler = self.lr_schedulers()
        #     if isinstance(scheduler, list):
        #         scheduler = scheduler[0]
            
        #     # Log learning rate of the optimizer
        #     for idx, param_group in enumerate(self.optimizers().param_groups):
        #         wandb.log({f"learning_rate_{idx}": param_group['lr']})
            
        #     # Log best metric value seen so far by the scheduler
        #     best_metric_val = scheduler.best
        #     wandb.log({"best_val_loss": best_metric_val})

        #     # Log number of epochs since last improvements
        #     epochs_since_improvement = scheduler.num_bad_epochs
        #     wandb.log({"epochs_since_improvement": epochs_since_improvement})
            
        #     # Manually step the scheduler
        #     scheduler.step(val_loss)
            
        # def test_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
        #     y_hat = self(x)
        #     loss_original = F.mse_loss(y_hat, y)
        #     y_hat_cam0 = self(x_cam0)
        #     loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
        #     y_hat_cam2 = self(x_cam2)
        #     loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
        #     y_hat_cam3 = self(x_cam3)
        #     loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
        #     avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
        #     wandb.log({"loss_test": avg_loss.item()})
        #     return avg_loss

        # def configure_optimizers(self):
        #     optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        #     scheduler = {
        #         'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.5),
        #         'monitor': 'loss_val',
        #         'interval': 'epoch',
        #         'frequency': 1
        #     }
        #     return [optimizer], [scheduler]

    class BehaveDataset(Dataset):
        def __init__(self, 
                    smpl_inputs_dist, smpl_reprojected_cam0_dist, smpl_reprojected_cam2_dist, smpl_reprojected_cam3_dist, 
                    obj_labels_dist, obj_reprojected_cam0_dist, obj_reprojected_cam2_dist, obj_reprojected_cam3_dist,
                    smpl_inputs_tri, smpl_reprojected_cam0_tri, smpl_reprojected_cam2_tri, smpl_reprojected_cam3_tri, 
                    obj_labels_tri, obj_reprojected_cam0_tri, obj_reprojected_cam2_tri, obj_reprojected_cam3_tri,
                    identifiers):

            # Data for distance regressor stage
            self.inputs_dist = smpl_inputs_dist
            self.reprojected_cam0_inputs_dist = smpl_reprojected_cam0_dist
            self.reprojected_cam2_inputs_dist = smpl_reprojected_cam2_dist
            self.reprojected_cam3_inputs_dist = smpl_reprojected_cam3_dist
            self.labels_dist = obj_labels_dist
            self.reprojected_cam0_labels_dist = obj_reprojected_cam0_dist
            self.reprojected_cam2_labels_dist = obj_reprojected_cam2_dist
            self.reprojected_cam3_labels_dist = obj_reprojected_cam3_dist
            
            # Data for trilateration stage
            self.inputs_tri = smpl_inputs_tri
            self.reprojected_cam0_inputs_tri = smpl_reprojected_cam0_tri
            self.reprojected_cam2_inputs_tri = smpl_reprojected_cam2_tri
            self.reprojected_cam3_inputs_tri = smpl_reprojected_cam3_tri
            self.labels_tri = obj_labels_tri
            self.reprojected_cam0_labels_tri = obj_reprojected_cam0_tri
            self.reprojected_cam2_labels_tri = obj_reprojected_cam2_tri
            self.reprojected_cam3_labels_tri = obj_reprojected_cam3_tri

            self.identifiers = identifiers

        def __len__(self):
            return len(self.inputs_dist)  # assuming same length for both stages

        def __getitem__(self, idx):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            return (
                torch.tensor(self.inputs_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam0_inputs_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam2_inputs_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam3_inputs_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.labels_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam0_labels_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam2_labels_dist[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam3_labels_dist[idx], dtype=torch.float32).to(device),

                torch.tensor(self.inputs_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam0_inputs_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam2_inputs_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam3_inputs_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.labels_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam0_labels_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam2_labels_tri[idx], dtype=torch.float32).to(device),
                torch.tensor(self.reprojected_cam3_labels_tri[idx], dtype=torch.float32).to(device),

                self.identifiers[idx]  # Assuming identifiers are not tensors; if they are, do the same as above
            )

    class BehaveDataModule(pl.LightningDataModule):
        def __init__(self, dataset, split, batch_size = wandb.config.batch_size):
            super(BehaveDataModule, self).__init__()
            self.dataset = dataset
            self.batch_size = batch_size
            self.split = split

            self.train_indices = []
            self.test_indices = []

            for idx, identifier in enumerate(self.dataset.identifiers):
                if identifier in self.split['train']:
                    self.train_indices.append(idx)
                elif identifier in self.split['test']:
                    self.test_indices.append(idx)

            #################
            # Using training set as validation
            self.test_indices = self.train_indices
            #################

        def train_dataloader(self):
            train_dataset = Subset(self.dataset, self.train_indices)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        def val_dataloader(self):
            # Assuming validation set is not provided, so using test set as validation
            test_dataset = Subset(self.dataset, self.test_indices)
            return DataLoader(test_dataset, batch_size=self.batch_size)

        def test_dataloader(self):
            test_dataset = Subset(self.dataset, self.test_indices)
            return DataLoader(test_dataset, batch_size=self.batch_size)

    class JointTransformerEncoder(nn.Module):
        def __init__(self, d_model, nhead, num_layers):
            super(JointTransformerEncoder, self).__init__()
            self.encoder_layer = TransformerEncoderLayer(d_model, nhead)
            self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)

        def forward(self, joints):
            output = self.transformer_encoder(joints)
            return output

    class CombinedMLP(pl.LightningModule):
        def __init__(self, input_dim, output_stage1, input_stage2, output_stage2, input_stage3, output_dim):
            super(CombinedMLP, self).__init__()

            # Two instances of the MLP model for two stages
            self.model1 = MLP1(input_dim, output_stage1)
            self.model2 = MLP_trilateration(input_stage2, output_stage2)
            self.model3 = MLP3(input_stage3,output_dim)
            self.automatic_optimization = False
            #self.relu = nn.ReLU()
            self.validation_losses = []
            self.best_avg_loss_val = float('inf')
            #self.joint_transformer = JointTransformerEncoder(d_model=72, nhead=2, num_layers=1)

        def forward(self, x, joints, obj):

            def normalize(tensor, min_range=0, max_range=1):
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
                normalized_tensor = (tensor - min_val) / (max_val - min_val) * (max_range - min_range) + min_range
                return normalized_tensor

            def gamma(p, L):
                # Ensure p is a tensor
                p = torch.tensor(p, dtype=torch.float32)
                
                # Create a range of frequencies
                frequencies = torch.arange(0, L).float()
                encodings = []
                
                # Compute the sine and cosine encodings
                for frequency in frequencies:
                    sin_encodings = torch.sin((2 ** (frequency)) * torch.pi * p)
                    cos_encodings = torch.cos((2 ** (frequency)) * torch.pi * p)
                    
                    # Interleave sin and cos encodings
                    encoding = torch.stack([sin_encodings, cos_encodings], dim=-1).reshape(-1)
                    encodings.append(encoding)
                
                # Concatenate all encodings
                return torch.cat(encodings, dim=-1)

            def encode(tensor, L):
                # Convert to a list of tensors for each batch
                batch_tensors = []
                for batch in tensor:
                    tensors = [gamma(p, L) for p in batch]
                    concatenated_tensor = torch.cat(tensors, dim=0)  # change from stack to cat
                    batch_tensors.append(concatenated_tensor)

                # Stack the tensors along the first dimension to create a new tensor
                final_tensor = torch.stack(batch_tensors)
                
                return final_tensor

            def axis_angle_to_rotation_matrix(axis_angle):
                device = axis_angle.device
                angle = torch.norm(axis_angle)
                if angle == 0:
                    return torch.eye(3).to(device)
                axis = axis_angle / angle
                cos_theta = torch.cos(angle)
                sin_theta = torch.sin(angle)
                cross_prod_matrix = torch.tensor([[0, -axis[2], axis[1]], 
                                                [axis[2], 0, -axis[0]], 
                                                [-axis[1], axis[0], 0]]).to(device)
                return torch.eye(3).to(device) * cos_theta + (1 - cos_theta) * torch.ger(axis, axis) + sin_theta * cross_prod_matrix

            def rotation_matrix_to_axis_angle(rot_matrix):
                epsilon = 1e-6
                device = rot_matrix.device
                if torch.allclose(rot_matrix, torch.eye(3).to(device), atol=epsilon):
                    return torch.zeros(3).to(device)
                cos_theta = (torch.trace(rot_matrix) - 1) / 2
                cos_theta = torch.clamp(cos_theta, -1, 1)
                theta = torch.acos(cos_theta)
                axis = torch.tensor([
                    rot_matrix[2, 1] - rot_matrix[1, 2],
                    rot_matrix[0, 2] - rot_matrix[2, 0],
                    rot_matrix[1, 0] - rot_matrix[0, 1]
                ]).to(device)
                norm = torch.norm(axis)
                if norm < epsilon:
                    return torch.zeros(3).to(device)
                return theta * axis / norm

            def process_pose_params(pose_params):
                device = pose_params.device
                pose_params = pose_params.view(-1, 3)
                parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).to(device)
                absolute_rotations_matrices = [torch.eye(3).to(device) for _ in range(len(parents))]

                def compute_absolute_rotation(joint_idx):
                    if parents[joint_idx] == -1:
                        return axis_angle_to_rotation_matrix(pose_params[joint_idx])
                    else:
                        parent_abs_rotation = compute_absolute_rotation(parents[joint_idx].item())
                        return torch.mm(parent_abs_rotation, axis_angle_to_rotation_matrix(pose_params[joint_idx]))

                for i in range(len(parents)):
                    absolute_rotations_matrices[i] = compute_absolute_rotation(i)

                absolute_rotations_axis_angle = [rotation_matrix_to_axis_angle(rot) for rot in absolute_rotations_matrices]
                return absolute_rotations_axis_angle

            # Function to convert axis-angle to relative rotations based on SMPL hierarchy
            def absolute_to_relative_rotations(absolute_rotations_axis_angle):
                parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
                
                relative_rotations = []
                
                for i, abs_rot in enumerate(absolute_rotations_axis_angle):
                    if parents[i] == -1:
                        relative_rotations.append(abs_rot)
                    else:
                        parent_abs_matrix = axis_angle_to_rotation_matrix(absolute_rotations_axis_angle[parents[i]])
                        joint_abs_matrix = axis_angle_to_rotation_matrix(abs_rot)
                        
                        # Compute relative rotation as: R_relative = R_parent_inverse * R_joint
                        relative_matrix = np.dot(np.linalg.inv(parent_abs_matrix), joint_abs_matrix)
                        relative_rotations.append(rotation_matrix_to_axis_angle(relative_matrix))
                
                return relative_rotations

            def process_pose_params_batch(batch_pose_params):
                # Get device info
                device = batch_pose_params.device
                
                # Ensure batch_pose_params is a PyTorch tensor on the same device
                batch_pose_params = torch.tensor(batch_pose_params, dtype=torch.float32).to(device)
                
                # Determine batch size and initialize list for results
                batch_size = batch_pose_params.shape[0]
                batch_results = []
                
                # Process each set of pose_params in the batch
                for b in range(batch_size):
                    pose_params = batch_pose_params[b].view(-1, 3)
                    absolute_rotations = process_pose_params(pose_params)
                    batch_results.append(torch.stack(absolute_rotations))  # Assuming absolute_rotations is a list of tensors
                
                # Stack results into a tensor and reshape
                batch_results_tensor = torch.stack(batch_results)
                
                return batch_results_tensor.view(batch_size, -1)
           
            #relative_rotations = absolute_to_relative_rotations(absolute_rotations)
            # Relative to absolute angles
            absolute_rotations = process_pose_params_batch(x[:,:72])

            # Normalization pose
            normalized_pose = normalize(absolute_rotations, 0, 2*torch.pi)
            #print("normalized_pose shape:", normalized_pose.shape)

            # Encoding pose (72 * L=3 * 2(sin/cos) = 432)
            encoded_pose = encode(normalized_pose, L=wandb.config.L)
            #print("encoded_pose shape:", encoded_pose.shape)
            x_stage1_pose = self.model1(encoded_pose)
            #print("x_stage1_pose shape:", x_stage1_pose.shape)

            # Normalization joints
            normalized_joints = normalize(joints[:,:72], 0, 2*np.pi)
            #print("normalized_joints shape:", normalized_joints.shape)

            # Encoding joints (72 * 3 * 2 = 432)
            encoded_joints = encode(normalized_joints, L=wandb.config.L)
            #print("encoded_joints shape:", encoded_joints.shape)
            x_stage1_joints = self.model1(encoded_joints)
            #print("x_stage1_joints shape:", x_stage1_joints.shape)

            concatenated_input = torch.cat((x_stage1_pose, x_stage1_joints), dim=1)
            #print("concatenated_input shape:", concatenated_input.shape)

            x_stage2 = self.model2(concatenated_input)
            #print("x_stage2 shape:", x_stage2.shape)

            #######
            #pos encoding 
            # x_stage2_enc = encode(x_stage2, L=wandb.config.L)
            # obj_enc = encode(obj, L=wandb.config.L)
            # x_stage3 = self.model3(torch.cat((x_stage2_enc,obj_enc), dim=-1))
            ########

            x_stage3 = self.model3(torch.cat((x_stage2,obj), dim=-1))
            #x_stage3 = torch.ones(obj.shape, requires_grad=True) * obj
            #print("x_stage3 shape:", x_stage3.shape)

            return x_stage1_pose,x_stage3

        def training_step(self, batch, batch_idx):
            x_cam1, x_cam0, x_cam2, x_cam3, y_cam1, y_cam0, y_cam2, y_cam3, x_cam1_stage2, x_cam0_stage2, x_cam2_stage2,\
            x_cam3_stage2, y_cam1_stage2, y_cam0_stage2, y_cam2_stage2, y_cam3_stage2, _ = batch

            def smooth_sign_loss(y, y_hat, alpha=wandb.config.alpha):
                # Clamp the product to prevent extreme values
                product = torch.clamp(alpha * y * y_hat, -10, 10)
                
                # Compute the stable loss
                loss = (1 - torch.sigmoid(product)).sum()
                
                return loss
            
            # Forward pass
            y_hat_cam0_stage1, y_hat_cam0_stage2 = self(x_cam0, x_cam0_stage2, y_cam0_stage2)
            y_hat_cam1_stage1, y_hat_cam1_stage2 = self(x_cam1, x_cam1_stage2, y_cam1_stage2)
            y_hat_cam2_stage1, y_hat_cam2_stage2 = self(x_cam2, x_cam2_stage2, y_cam2_stage2)
            y_hat_cam3_stage1, y_hat_cam3_stage2 = self(x_cam3, x_cam3_stage2, y_cam3_stage2)

            # Compute the losses for the first stage
            # loss_original_stage1 = F.mse_loss(y_hat_cam1_stage1, y) + sign_sensitive_loss(y_hat_cam1_stage1, y)
            # loss_cam0_stage1 = F.mse_loss(y_hat_cam0_stage1, y_cam0) + sign_sensitive_loss(y_hat_cam0_stage1, y_cam0)
            # loss_cam2_stage1 = F.mse_loss(y_hat_cam2_stage1, y_cam2) + sign_sensitive_loss(y_hat_cam2_stage1, y_cam2)
            # loss_cam3_stage1 = F.mse_loss(y_hat_cam3_stage1, y_cam3) + sign_sensitive_loss(y_hat_cam3_stage1, y_cam3)

            # Define weights
            lambda_1 = wandb.config.lambda_1
            lambda_2 = wandb.config.lambda_2

            # Original stage2
            # loss_cam1_stage2 = lambda_1 * F.mse_loss(y_hat_cam1_stage2, y_cam1_stage2) + lambda_2 * smooth_sign_loss(y_hat_cam1_stage2, y_cam1_stage2)
            # loss_cam0_stage2 = lambda_1 * F.mse_loss(y_hat_cam0_stage2, y_cam0_stage2) + lambda_2 * smooth_sign_loss(y_hat_cam0_stage2, y_cam0_stage2)
            # loss_cam3_stage2 = lambda_1 * F.mse_loss(y_hat_cam3_stage2, y_cam3_stage2) + lambda_2 * smooth_sign_loss(y_hat_cam3_stage2, y_cam3_stage2)
            # loss_cam2_stage2 = lambda_1 * F.mse_loss(y_hat_cam2_stage2, y_cam2_stage2) + lambda_2 * smooth_sign_loss(y_hat_cam2_stage2, y_cam2_stage2)
            ############################################### Extract the first two components for y_hat and y

            # # Train only first two components
            # y_hat_cam1_stage2_first_two = y_hat_cam1_stage2[:, :2]
            # y_cam1_stage2_first_two = y_cam1_stage2[:, :2]

            # y_hat_cam0_stage2_first_two = y_hat_cam0_stage2[:, :2]
            # y_cam0_stage2_first_two = y_cam0_stage2[:, :2]

            # y_hat_cam3_stage2_first_two = y_hat_cam3_stage2[:, :2]
            # y_cam3_stage2_first_two = y_cam3_stage2[:, :2]

            # y_hat_cam2_stage2_first_two = y_hat_cam2_stage2[:, :2]
            # y_cam2_stage2_first_two = y_cam2_stage2[:, :2]

            # # Calculate the modified loss for each camera
            # loss_cam1_stage2 = lambda_1 * F.mse_loss(y_hat_cam1_stage2_first_two, y_cam1_stage2_first_two) + lambda_2 * smooth_sign_loss(y_hat_cam1_stage2_first_two, y_cam1_stage2_first_two)
            # loss_cam0_stage2 = lambda_1 * F.mse_loss(y_hat_cam0_stage2_first_two, y_cam0_stage2_first_two) + lambda_2 * smooth_sign_loss(y_hat_cam0_stage2_first_two, y_cam0_stage2_first_two)
            # loss_cam3_stage2 = lambda_1 * F.mse_loss(y_hat_cam3_stage2_first_two, y_cam3_stage2_first_two) + lambda_2 * smooth_sign_loss(y_hat_cam3_stage2_first_two, y_cam3_stage2_first_two)
            # loss_cam2_stage2 = lambda_1 * F.mse_loss(y_hat_cam2_stage2_first_two, y_cam2_stage2_first_two) + lambda_2 * smooth_sign_loss(y_hat_cam2_stage2_first_two, y_cam2_stage2_first_two)
            #################################################
            # Loss for camera 1
            loss_cam1_stage2 = lambda_1 * F.mse_loss(y_hat_cam1_stage2, y_cam1_stage2) + lambda_1 * (1 - F.cosine_similarity(y_hat_cam1_stage2, y_cam1_stage2)) + lambda_2 * smooth_sign_loss(y_hat_cam1_stage2, y_cam1_stage2)
            # print("loss_cam1_stage2:", loss_cam1_stage2)
            # print("y_hat_cam1_stage2:", y_hat_cam1_stage2)
            # print("y_cam1_stage2:", y_cam1_stage2)

            # Loss for camera 0
            loss_cam0_stage2 = lambda_1 * F.mse_loss(y_hat_cam0_stage2, y_cam0_stage2) + lambda_1 * (1 - F.cosine_similarity(y_hat_cam0_stage2, y_cam0_stage2)) + lambda_2 * smooth_sign_loss(y_hat_cam0_stage2, y_cam0_stage2)
            # print("loss_cam0_stage2:", loss_cam0_stage2)
            # print("y_hat_cam0_stage2:", y_hat_cam0_stage2)
            # print("y_cam0_stage2:", y_cam0_stage2)

            # Loss for camera 3
            loss_cam3_stage2 = lambda_1 * F.mse_loss(y_hat_cam3_stage2, y_cam3_stage2) + lambda_1 * (1 - F.cosine_similarity(y_hat_cam3_stage2, y_cam3_stage2)) + lambda_2 * smooth_sign_loss(y_hat_cam3_stage2, y_cam3_stage2)
            # print("loss_cam3_stage2:", loss_cam3_stage2)
            # print("y_hat_cam3_stage2:", y_hat_cam3_stage2)
            # print("y_cam3_stage2:", y_cam3_stage2)

            # Loss for camera 2
            loss_cam2_stage2 = lambda_1 * F.mse_loss(y_hat_cam2_stage2, y_cam2_stage2) + lambda_1 * (1 - F.cosine_similarity(y_hat_cam2_stage2, y_cam2_stage2)) + lambda_2 * smooth_sign_loss(y_hat_cam2_stage2, y_cam2_stage2)
            # print("loss_cam2_stage2:", loss_cam2_stage2)
            # print("y_hat_cam2_stage2:", y_hat_cam2_stage2)
            # print("y_cam2_stage2:", y_cam2_stage2)
      
            total_loss = (loss_cam1_stage2 + loss_cam0_stage2 + loss_cam2_stage2 + loss_cam3_stage2)**2
            # print("total_loss:", total_loss)

            avg_loss = total_loss.mean()
            # print("avg_loss:", avg_loss)


            # print("Shape of loss_cam1_stage2:", loss_cam1_stage2.shape)
            # print("Shape of total_loss:", total_loss.shape)
            # print("Shape of avg_loss:", avg_loss.shape)

            # Log the individual and average losses to wandb
            wandb.log({
                "loss_train_sum_4_cams": avg_loss.item(),
                #"loss_original_stage1": loss_original_stage1.item(),
                #"loss_cam1_stage2": loss_cam1_stage2.item(),
                # Add other individual losses if needed
            })

            self.validation_losses.append(avg_loss)
            
            # Backward pass and optimization
            self.manual_backward(avg_loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

            return avg_loss

        # def training_epoch_end(self, training_step_outputs):
        #     # Get current time for unique filename
        #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #     print("Hello")
        #     # Save the model at the end of the epoch with a unique name
        #     filename = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/pos_enc/{wandb.run.name}_{timestamp}.pt"
        #     torch.save(model_combined, filename)
        #     model = torch.load(filename)
        #     # Set the model to evaluation mode
        #     model.eval()
            
        #     total_mse = []  # Initialize as empty list
        #     cameras = ['cam0', 'cam1', 'cam2', 'cam3']

        #     for cam in cameras:
        #         # Extract relevant data for the current camera
        #         reprojected_smpl_dist = globals()[f'reprojected_smpl_{cam}_dist']
        #         reprojected_smpl_tri = globals()[f'reprojected_smpl_{cam}_tri']
        #         reprojected_obj_tri = globals()[f'reprojected_obj_{cam}_tri']
                
        #         for smpl, joints, obj in zip(reprojected_smpl_dist, reprojected_smpl_tri, reprojected_obj_tri):
        #             input_tensor = torch.tensor(smpl, dtype=torch.float32)
        #             input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
        #             selected_joints_tensor = torch.tensor(joints, dtype=torch.float32).unsqueeze(0)
        #             obj_trans_GT = torch.tensor(obj, dtype=torch.float32).unsqueeze(0)
        #             candidate_pos = model(input_tensor, selected_joints_tensor, obj_trans_GT)
        #             obj_trans_vector = candidate_pos[1].detach().numpy().flatten()

        #             def log_mse_to_wandb(true_vector, predicted_vector):
        #                 # Compute the squared differences
        #                 squared_diffs = (true_vector - predicted_vector) ** 2
                        
        #                 # Calculate the mean squared error
        #                 mse = np.mean(squared_diffs)
                        
        #                 # Log the MSE to wandb
        #                 wandb.log({f"MSE_{cam}": mse})

        #                 return mse  # Return the mse

        #             mse = log_mse_to_wandb(obj_trans_vector, obj)  # Get the returned mse
        #             total_mse.append(mse)  # Append the mse to the list

        #     # Convert the list to a Torch tensor and compute the mean
        #     mean_mse = torch.tensor(total_mse).mean().item()  # Convert mean_mse to a Python scalar using .item()

        #     # Log the mean MSE
        #     wandb.log({"mean_MSE": mean_mse})

        #     # Optionally, return the log
        #     return {"mean_MSE": mean_mse}


        def validation_step(self, batch, batch_idx):
            x_cam1, x_cam0, x_cam2, x_cam3, y_cam1, y_cam0, y_cam2, y_cam3, x_cam1_stage2, x_cam0_stage2, x_cam2_stage2,\
            x_cam3_stage2, y_cam1_stage2, y_cam0_stage2, y_cam2_stage2, y_cam3_stage2, _ = batch
            
            # Forward pass
            y_hat_cam0_stage1, y_hat_cam0_stage2 = self(x_cam0, x_cam0_stage2, y_cam0_stage2)
            y_hat_cam1_stage1, y_hat_cam1_stage2 = self(x_cam1, x_cam1_stage2, y_cam1_stage2)
            y_hat_cam2_stage1, y_hat_cam2_stage2 = self(x_cam2, x_cam2_stage2, y_cam2_stage2)
            y_hat_cam3_stage1, y_hat_cam3_stage2 = self(x_cam3, x_cam3_stage2, y_cam3_stage2)


            # Loss
            loss_cam1_stage2 = F.mse_loss(y_hat_cam1_stage2, y_cam1_stage2)
            loss_cam0_stage2 = F.mse_loss(y_hat_cam0_stage2, y_cam0_stage2)
            loss_cam3_stage2 = F.mse_loss(y_hat_cam3_stage2, y_cam3_stage2)
            loss_cam2_stage2 = F.mse_loss(y_hat_cam2_stage2, y_cam2_stage2)
            avg_loss = (loss_cam1_stage2 + loss_cam0_stage2 + loss_cam2_stage2 + loss_cam3_stage2)/4

            self.log('val_loss', avg_loss, prog_bar=False, logger=False)

            self.validation_losses.append(avg_loss)
            return {'val_loss': avg_loss}
        
        def on_validation_epoch_end(self):
            avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
            wandb.log({"avg_loss_val": avg_val_loss.item()})
            self.log('avg_val_loss', avg_val_loss, prog_bar=True, logger=True)
            wandb.log({"learning rate": self.optimizer.param_groups[0]['lr']})
            if avg_val_loss < self.best_avg_loss_val:
                self.best_avg_loss_val = avg_val_loss
                print(f"Best number of epochs:{self.current_epoch}")

            self.validation_losses = []  # reset for the next epoch
            self.lr_scheduler.step(avg_val_loss)  # Up

            # Save the model
            model_save_path = f'/scratch_net/biwidl307/lgermano/H2O/trained_models/pos_enc/model_{wandb.run.name}_epoch_{self.current_epoch}.pt'
            torch.save(self.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')


        # def test_step(self, batch, batch_idx):
        #     x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, x_cam1_stage2, x_cam0_stage2, x_cam2_stage2, x_cam3_stage2, y_cam1_stage2, y_cam0_stage2, y_cam2_stage2, y_cam3_stage2, _ = batch
        #     y_hat_stage1, y_hat_stage2 = self(x)
            
        #     # Compute test losses for both stages
        #     loss_original_stage1 = F.mse_loss(y_hat_stage1, y)
        #     loss_cam1_stage2 = F.mse_loss(y_hat_stage2, y_cam1_stage2)
        #     avg_loss = (loss_original_stage1 + loss_cam1_stage2) / 2

        #     wandb.log({"loss_test": avg_loss.item()})
        #     # Log the individual losses for testing to wandb
        #     wandb.log({
        #         "loss_test_stage1": loss_original_stage1.item(),
        #         "loss_test_stage2": loss_cam1_stage2.item()
        #     })
        #     return avg_loss

        def configure_optimizers(self):
            if wandb.config.optimizer == "SGD":
                optimizer = torch.optim.SGD(self.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=1e-4)
            elif wandb.config.optimizer == "Adagrad":
                optimizer = torch.optim.Adagrad(self.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)
            elif wandb.config.optimizer == "RMSprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=wandb.config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=1e-4, momentum=0.9)
            elif wandb.config.optimizer == "AdamW":
                optimizer = torch.optim.AdamW(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
            elif wandb.config.optimizer == "Adadelta":
                optimizer = torch.optim.Adadelta(self.parameters(), lr=wandb.config.learning_rate, rho=0.9, eps=1e-06, weight_decay=1e-4)
            elif wandb.config.optimizer == "LBFGS":
                optimizer = torch.optim.LBFGS(self.parameters(), lr=wandb.config.learning_rate, max_iter=20, line_search_fn='strong_wolfe')
            else:  # default to Adam if no match
                optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

            # scheduler = {
            #     'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True, factor=0.01, threshold=0.75, threshold_mode='rel'),
            #     'monitor': 'avg_val_loss',
            #     'interval': 'epoch',
            #     'frequency': 1
            # }

            scheduler = {
            'scheduler': CustomCyclicLR(optimizer, base_lr=1e-7, max_lr=1e-3, step_size=1, mode='exp_range'),
            'interval': 'step',  # step-based updates i.e. batch
            #'monitor' : 'avg_val_loss',
            'name': 'custom_clr'
            }

            self.optimizer = optimizer  # store optimizer as class variable for logging learning rate
            self.lr_scheduler = scheduler['scheduler']  # store scheduler as class variable for updating in on_validation_epoch_end
            return [optimizer], [scheduler]
   
    ####################################################################################
    # 4. Training using PyTorch Lightnings
    # Integrating the loading and dataset creation
    behave_seq = "/scratch_net/biwidl307_second/lgermano/behave/sequences/Date07_Sub04_yogaball_play"
    base_path = "/scratch_net/biwidl307_second/lgermano/behave"

    data_file_path = '/scratch_net/biwidl307/lgermano/H2O/datasets/MLP_combined/Date07_Sub04_yogaball_play_dataset.pkl'

    # Check if the data has already been saved
    if os.path.exists(data_file_path):
        # Load the saved data
        with open(data_file_path, 'rb') as f:
            data_retrieved = pickle.load(f)
            (reprojected_smpl_cam1_dist, reprojected_smpl_cam0_dist, reprojected_smpl_cam2_dist, reprojected_smpl_cam3_dist,
            reprojected_obj_cam1_dist, reprojected_obj_cam0_dist, reprojected_obj_cam2_dist, reprojected_obj_cam3_dist, 
            identifiers_dist, reprojected_smpl_cam1_tri, reprojected_smpl_cam0_tri, reprojected_smpl_cam2_tri, 
            reprojected_smpl_cam3_tri, reprojected_obj_cam1_tri, reprojected_obj_cam0_tri, reprojected_obj_cam2_tri, 
            reprojected_obj_cam3_tri, identifiers_tri) = data_retrieved
    else:

        ############## USING A SUBSET ######################
        all_files = sorted(glob.glob(os.path.join(base_path, "sequences", "Date07_Sub04_yogaball_play", "t*.000")))
        selected_files = all_files
    
        #print(f"Detected {len(selected_files)} frames.")

        all_data_frames = []

        # Gather data into all_data_frames
        for idx, frame_folder in enumerate(selected_files):
            frame_data = {}

            frame_data['smpl_path'] = os.path.join(frame_folder, "person", "fit02", "person_fit.pkl")
            object_name = "sports ball"
            frame_data['obj_path'] = os.path.join(frame_folder, object_name, "fit01", f"{object_name}_fit.pkl")
            frame_data['scene'] = os.path.basename(frame_folder)

            
            smpl_data = load_pickle(frame_data['smpl_path'])
            frame_data['pose'] = smpl_data['pose']
            frame_data['trans'] = smpl_data['trans']
            frame_data['betas'] = smpl_data['betas']

            obj_data = load_pickle(frame_data['obj_path'])
            frame_data['obj_pose'] = obj_data['angle']
            frame_data['obj_trans'] = obj_data['trans']
            
            image_paths = {
                1: os.path.join(frame_folder, "k1.color.jpg"),
                2: os.path.join(frame_folder, "k2.color.jpg"),
                0: os.path.join(frame_folder, "k0.color.jpg"),
                3: os.path.join(frame_folder, "k3.color.jpg")
            }
                        
            frame_data['img'] = image_paths
                
            all_data_frames.append(frame_data)

        # Interpolate between frames

        interpolated_data_frames = interpolate_frames(all_data_frames)

        # Load data for distance regressor stage
        reprojected_smpl_cam1_dist, reprojected_smpl_cam0_dist, reprojected_smpl_cam2_dist, reprojected_smpl_cam3_dist, \
        reprojected_obj_cam1_dist, reprojected_obj_cam0_dist, reprojected_obj_cam2_dist, reprojected_obj_cam3_dist, \
        identifiers_dist = load_frames_distance_regressor(interpolated_data_frames)

        # Load data for trilateration stage
        reprojected_smpl_cam1_tri, reprojected_smpl_cam0_tri, reprojected_smpl_cam2_tri, reprojected_smpl_cam3_tri, \
        reprojected_obj_cam1_tri, reprojected_obj_cam0_tri, reprojected_obj_cam2_tri, reprojected_obj_cam3_tri, \
        identifiers_tri = load_frames_trilateration(interpolated_data_frames)

        # Save the data for future use
        with open(data_file_path, 'wb') as f:
            data_to_save = (reprojected_smpl_cam1_dist, reprojected_smpl_cam0_dist, reprojected_smpl_cam2_dist, reprojected_smpl_cam3_dist,
                            reprojected_obj_cam1_dist, reprojected_obj_cam0_dist, reprojected_obj_cam2_dist, reprojected_obj_cam3_dist, 
                            identifiers_dist, reprojected_smpl_cam1_tri, reprojected_smpl_cam0_tri, reprojected_smpl_cam2_tri, 
                            reprojected_smpl_cam3_tri, reprojected_obj_cam1_tri, reprojected_obj_cam0_tri, reprojected_obj_cam2_tri, 
                            reprojected_obj_cam3_tri, identifiers_tri)
            pickle.dump(data_to_save, f)

    # Ensure that identifiers are the same (or however you wish to verify data consistency)
    assert identifiers_dist == identifiers_tri
     
    input_dim = 72 * wandb.config.L * 2
    output_stage1 = 256
    input_stage2 = 512
    output_stage2 = 3
    input_stage3 = 6 # 2 * (3 * wandb.config.L * 2)
    output_dim = 3

    print(f"reprojected_smpl_cam1_dist length: {len(reprojected_smpl_cam1_dist)}")
    print(f"reprojected_smpl_cam0_dist length: {len(reprojected_smpl_cam0_dist)}")
    print(f"reprojected_smpl_cam2_dist length: {len(reprojected_smpl_cam2_dist)}")
    print(f"reprojected_smpl_cam3_dist length: {len(reprojected_smpl_cam3_dist)}")
    print(f"reprojected_obj_cam1_dist length: {len(reprojected_obj_cam1_dist)}")
    print(f"reprojected_obj_cam0_dist length: {len(reprojected_obj_cam0_dist)}")
    print(f"reprojected_obj_cam2_dist length: {len(reprojected_obj_cam2_dist)}")
    print(f"reprojected_obj_cam3_dist length: {len(reprojected_obj_cam3_dist)}")
    print(f"identifiers_dist length: {len(identifiers_dist)}")
    print(f"reprojected_smpl_cam1_tri length: {len(reprojected_smpl_cam1_tri)}")
    print(f"reprojected_smpl_cam0_tri length: {len(reprojected_smpl_cam0_tri)}")
    print(f"reprojected_smpl_cam2_tri length: {len(reprojected_smpl_cam2_tri)}")
    print(f"reprojected_smpl_cam3_tri length: {len(reprojected_smpl_cam3_tri)}")
    print(f"reprojected_obj_cam1_tri length: {len(reprojected_obj_cam1_tri)}")
    print(f"reprojected_obj_cam0_tri length: {len(reprojected_obj_cam0_tri)}")
    print(f"reprojected_obj_cam2_tri length: {len(reprojected_obj_cam2_tri)}")
    print(f"reprojected_obj_cam3_tri length: {len(reprojected_obj_cam3_tri)}")
    print(f"identifiers_tri length: {len(identifiers_tri)}")

    print(f"Wandb run name:{wandb.run.name}")


    ############OFFSET########################

    def offset_transform(input_list, identifiers):
        offsetted_list = []

        for idx, vector in enumerate(input_list[:-1]):
            if identifiers[idx] == identifiers[idx + 1] and idx >=1:
                vector_offsetted = vector.copy()
                vector_offsetted -= input_list[idx-1]
                offsetted_list.append(vector_offsetted)
            else:
                offsetted_list.append(vector)

        offsetted_list.append(input_list[-1])
        return offsetted_list

    reprojected_smpl_cam1_dist = offset_transform(reprojected_smpl_cam1_dist, identifiers_dist)
    reprojected_smpl_cam0_dist = offset_transform(reprojected_smpl_cam0_dist, identifiers_dist)
    reprojected_smpl_cam2_dist = offset_transform(reprojected_smpl_cam2_dist, identifiers_dist)
    reprojected_smpl_cam3_dist = offset_transform(reprojected_smpl_cam3_dist, identifiers_dist)
    reprojected_obj_cam1_dist = offset_transform(reprojected_obj_cam1_dist, identifiers_dist)
    reprojected_obj_cam0_dist = offset_transform(reprojected_obj_cam0_dist, identifiers_dist)
    reprojected_obj_cam2_dist = offset_transform(reprojected_obj_cam2_dist, identifiers_dist)
    reprojected_obj_cam3_dist = offset_transform(reprojected_obj_cam3_dist, identifiers_dist)
    # For 'identifiers_dist', there's no corresponding line in the given print statements
    reprojected_smpl_cam1_tri = offset_transform(reprojected_smpl_cam1_tri, identifiers_tri)
    reprojected_smpl_cam0_tri = offset_transform(reprojected_smpl_cam0_tri, identifiers_tri)
    reprojected_smpl_cam2_tri = offset_transform(reprojected_smpl_cam2_tri, identifiers_tri)
    reprojected_smpl_cam3_tri = offset_transform(reprojected_smpl_cam3_tri, identifiers_tri)
    reprojected_obj_cam1_tri = offset_transform(reprojected_obj_cam1_tri, identifiers_tri)
    reprojected_obj_cam0_tri = offset_transform(reprojected_obj_cam0_tri, identifiers_tri)
    reprojected_obj_cam2_tri = offset_transform(reprojected_obj_cam2_tri, identifiers_tri)
    reprojected_obj_cam3_tri = offset_transform(reprojected_obj_cam3_tri, identifiers_tri)
    # For 'identifiers_tri', there's no corresponding line in the given print statements

    ##########################################

    dataset = BehaveDataset(
        reprojected_smpl_cam1_dist,
        reprojected_smpl_cam0_dist,
        reprojected_smpl_cam2_dist,
        reprojected_smpl_cam3_dist,
        reprojected_obj_cam1_dist,
        reprojected_obj_cam0_dist,
        reprojected_obj_cam2_dist,
        reprojected_obj_cam3_dist,
        reprojected_smpl_cam1_tri,
        reprojected_smpl_cam0_tri,
        reprojected_smpl_cam2_tri,
        reprojected_smpl_cam3_tri,
        reprojected_obj_cam1_tri,
        reprojected_obj_cam0_tri,
        reprojected_obj_cam2_tri,
        reprojected_obj_cam3_tri,
        identifiers_dist  # or identifiers_tri, since they are asserted to be the same
    )


    ############# CHECK THE 4 CAMERAS LABELS HAVE SAME DISTRIBUTION ###########

    # # Sample data (replace with your actual data)
    # reprojected_obj_data = [
    #     reprojected_obj_cam0_tri,
    #     reprojected_obj_cam1_tri,
    #     reprojected_obj_cam2_tri,
    #     reprojected_obj_cam3_tri,
    # ]

    # # Create a figure and axis
    # fig, ax = plt.subplots(figsize=(10, 6))


    # # Iterate through the list of lists and plot the three components of object labels for each camera
    # for idx, camera_data in enumerate(reprojected_obj_data):
    #     camera_name = f'cam{idx}'  # Extract the camera name based on the list index
    #     for component in range(3):
    #         component_data = np.array([item[component] for item in camera_data]).flatten()
    #         ax.hist(
    #             component_data,
    #             bins=30,
    #             alpha=0.5,
    #             label=f'{camera_name} Component {component + 1}'
    #         )

    #     # Add labels and legend
    #     ax.set_xlabel('Value')
    #     ax.set_ylabel('Frequency')
    #     ax.set_title('Distribution of Object Label Components for Each Camera')
    #     ax.legend()

    # # Save the plot as an image at the given path
    # save_path = '/scratch_net/biwidl307/lgermano/crossvit/visualizations/plot_labels_distribution.png' 
    # plt.savefig(save_path)

    # # Optional: Close the plot to free up resources
    # plt.close()

    #######################################################
    ################### PLOT SIGN

    # Sample data (replace with your actual data)
    # reprojected_obj_data = [
    #     reprojected_obj_cam0_tri,
    #     reprojected_obj_cam1_tri,
    #     reprojected_obj_cam2_tri,
    #     reprojected_obj_cam3_tri,
    # ]

    # # Define colors for positive, negative, and zero signs
    # colors = ['blue', 'red', 'green']

    # # Create a single figure to contain all subplots
    # fig, axs = plt.subplots(3, len(reprojected_obj_data), figsize=(15, 10))

    # # Iterate through the list of lists and create separate plots for each component and camera
    # for idx, camera_data in enumerate(reprojected_obj_data):
    #     camera_name = f'cam{idx}'  # Extract the camera name based on the list index
    #     for component in range(3):
    #         component_data = np.array([item[component] for item in camera_data]).flatten()
    #         component_sign = np.sign(component_data).astype(int)  # Convert to integer

    #         # Plot histograms of the sign statistics with different colors and some separation
    #         bins = [-1.5, -0.5, 0.5, 1.5]  # Adjusted bin positions and widths
    #         axs[component, idx].hist(
    #             component_sign,
    #             bins=bins,
    #             alpha=0.5,
    #             label=f'{camera_name} Component {component + 1}',
    #             color=colors[component_sign[0] + 1]  # Map sign to color
    #         )

    #         # Add labels and legend for each subplot
    #         axs[component, idx].set_xlabel('Sign')
    #         axs[component, idx].set_ylabel('Frequency')
    #         axs[component, idx].set_title(f'Sign Statistics of Component {component + 1}')
    #         axs[component, idx].legend()

    # # Adjust layout and spacing between subplots
    # plt.tight_layout()

    # # Save the entire figure as an image at a specific path
    # save_path = '/scratch_net/biwidl307/lgermano/crossvit/visualizations/sign_statistics_combined.png'
    # plt.savefig(save_path)

    # # Optional: Close the plot to free up resources
    # plt.close()

    ######################################################################
    # dataset = BehaveDataset(
    #     reprojected_smpl_cam3_dist,
    #     reprojected_smpl_cam3_dist,
    #     reprojected_smpl_cam3_dist,
    #     reprojected_smpl_cam3_dist,
    #     reprojected_obj_cam3_dist,
    #     reprojected_obj_cam3_dist,
    #     reprojected_obj_cam3_dist,
    #     reprojected_obj_cam3_dist,
    #     reprojected_smpl_cam3_tri,
    #     reprojected_smpl_cam3_tri,
    #     reprojected_smpl_cam3_tri,
    #     reprojected_smpl_cam3_tri,
    #     reprojected_obj_cam3_tri,
    #     reprojected_obj_cam3_tri,
    #     reprojected_obj_cam3_tri,
    #     reprojected_obj_cam3_tri,
    #     identifiers_dist  # or identifiers_tri, since they are asserted to be the same
    # )

    path_to_file = "/scratch_net/biwidl307_second/lgermano/behave/split.json"
    split_dict = load_split_from_path(path_to_file)

    # Train and validate your model with the current set of hyperparameters
    data_module = BehaveDataModule(dataset, split=split_dict, batch_size=BATCH_SIZE)

    train_size = len(data_module.train_indices)
    # Assuming validation set is not provided, so using test set as validation
    val_size = len(data_module.test_indices)
    test_size = len(data_module.test_indices)

    print(f"Size of train set: {train_size}")
    print(f"Size of val set: {val_size}")
    print(f"Size of test set: {test_size}")
    
    # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on: {device}")

    # Initialize model
    model_combined = CombinedMLP(input_dim, output_stage1, input_stage2, output_stage2, input_stage3, output_dim)

    # Move the model to device
    model_combined.to(device)

    # Specify the path to the checkpoint
    # model_path = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/pos_enc/model_flowing-shadow-2137_epoch_44.pt"

    #Load the state dict from the checkpoint into the model
    # checkpoint = torch.load(model_path, map_location=device)
    # model_combined.load_state_dict(checkpoint)

    # Set the model to evaluation mode (you might want to set it to evaluation mode with `.eval()` if you're not training)
    model_combined.train()

    # Initialize Trainer
    trainer = pl.Trainer(max_epochs=wandb.config.epochs, num_sanity_val_steps=0, gpus=1 if torch.cuda.is_available() else 0)

    # Fit the model
    trainer.fit(model_combined, datamodule=data_module)

    # Adjusted computation for average validation loss
    if model_combined.best_avg_loss_val < best_overall_avg_loss_val:
        best_overall_avg_loss_val = model_combined.best_avg_loss_val

        best_params = {
            "learning_rate": LEARNING_RATE,
            "architecture": "MLP",
            "dataset": "BEHAVE",
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "layer_sizes_1": LAYER_SIZES_1,
            "layer_sizes_3": LAYER_SIZES_3,
            "alpha": ALPHA,
            "lambda_1": LAMBDA_1,
            "lambda_2": LAMBDA_2,
            "L": L,
            "epochs": EPOCHS
        }

        # # Optionally, to test the model:
        # trainer.test(combined_model, datamodule=data_module)

        # Save the model using WandB run ID
        filename = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/pos_enc/{wandb.run.name}.pt"

        # Save the model
        torch.save(model_combined, filename)

    # Finish the current W&B run
    wandb.finish()

#After all trials, #print the best set of hyperparameters
print("Best Validation Loss:", best_overall_avg_loss_val)
print("Best Hyperparameters:", best_params)