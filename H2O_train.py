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

best_overall_avg_loss_val = float('inf')
best_params = None

# Set the WANDB_CACHE_DIR environment variable
os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"

#learning_rate_range = [1e-3]#, 1e-1]#, 1e-4, 1e-5]
epochs_range = [5000]#,300]
learning_rate_range =  [1e-3]#, 5e-7, 1e-6, 5e-6, 1e-5]#[1e-6]#, 5e-3, 1e-2, 5e-2]
batch_size_range = [64]# whole training set #[32768]#[4096]#[32, 64, 128]
dropout_rate_range = [0.07]#, 0.1, 0.3]
alpha_range = [1]
lambda_1_range = [1]
lambda_2_range = [1]
lambda_3_range = [1]
lambda_4_range = [1]
L_range = [4]#,6,10]
optimizer_list =  ["AdamW"]#, "Adagrad","AdamW", "Adadelta", "LBFGS"]#, "Adam"]#["Adagrad", "RMSprop", "AdamW", "Adadelta", "LBFGS", "Adam"]
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

for lr, bs, dr, layers_1, layers_3, alpha, lambda_1, lambda_2, lambda_3, lambda_4, l, epochs, optimizer_name in itertools.product(
    learning_rate_range, batch_size_range, dropout_rate_range, layer_sizes_range_1, layer_sizes_range_3, alpha_range, lambda_1_range, lambda_2_range, lambda_3_range, lambda_4_range, L_range, epochs_range, optimizer_list
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
    LAMBDA_3 = lambda_3
    LAMBDA_4 = lambda_4
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
            "lambda_3": LAMBDA_3,
            "lambda_4": LAMBDA_4,
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
        # Number of frames after = 1 + (N-1) * len(range)
        interpolated_frames = []

        for idx in range(len(all_data_frames)-1):
            frame1 = all_data_frames[idx]
            frame2 = all_data_frames[idx+1]

            # Original frame
            interpolated_frames.append(frame1)

            # Interpolated frames

            for i in range(6):
                interpolated_frame = copy.deepcopy(frame1)
                t = i / 6.0  # Assuming you want to interpolate at 1/3 and 2/3 positions between frame1 and frame2
                interpolated_frame['pose'] = slerp_rotations(frame1['pose'], frame2['pose'], t)
                interpolated_frame['trans'] = linear_interpolate(frame1['trans'], frame2['trans'], t)
                interpolated_frame['obj_pose'] = slerp_rotations(frame1['obj_pose'], frame2['obj_pose'], t)
                interpolated_frame['obj_trans'] = linear_interpolate(frame1['obj_trans'], frame2['obj_trans'], t)
                
                interpolated_frames.append(interpolated_frame)            

        # Adding the last original frame
        interpolated_frames.append(all_data_frames[-1])

        return interpolated_frames

    def transform_smpl_to_camera_frame(pose, trans, cam_params):
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
        return transformed_pose, transformed_trans

    def transform_object_to_camera_frame(obj_pose, obj_trans, cam_params):
        """Transform object's position and orientation to another camera frame using relative transformation."""
        # Convert the axis-angle rotation to a matrix

        R_w = Rotation.from_rotvec(obj_pose).as_matrix()

        # Build transformation matrix of mesh in world coordinates
        T_mesh = np.eye(4)
        T_mesh[:3, :3] = R_w
        T_mesh[:3, 3] = obj_trans
        
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
        transformed_pose = Rotation.from_matrix(T_mesh_in_cam[:3, :3]).as_rotvec().flatten()

        return transformed_pose, transformed_trans
    
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

    def normalize(vec, min_range, max_range, min_val, max_val):
        normalized_vec = (vec - min_val) / (max_val - min_val) * (max_range - min_range) + min_range
        return normalized_vec

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


    # Function to convert axis-angle to rotation matrix
    def axis_angle_to_rotation_matrix(axis_angle):
        angle = np.linalg.norm(axis_angle)
        if angle == 0:
            return np.eye(3)
        axis = axis_angle / angle
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        cross_prod_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis, axis) + sin_theta * cross_prod_matrix

    # Function to convert rotation matrix to axis-angle
    def rotation_matrix_to_axis_angle(rot_matrix):
        epsilon = 1e-6  # threshold to handle numerical imprecisions
        if np.allclose(rot_matrix, np.eye(3), atol=epsilon):
            return np.zeros(3)  # no rotation

        # Extract the rotation angle
        cos_theta = (np.trace(rot_matrix) - 1) / 2
        cos_theta = np.clip(cos_theta, -1, 1)  # handle potential numerical issues
        theta = np.arccos(cos_theta)

        # Extract the rotation axis
        axis = np.array([
            rot_matrix[2, 1] - rot_matrix[1, 2],
            rot_matrix[0, 2] - rot_matrix[2, 0],
            rot_matrix[1, 0] - rot_matrix[0, 1]
        ])

        norm = np.linalg.norm(axis)
        if norm < epsilon:  # handle edge case
            return np.zeros(3)

        return theta * axis / norm
    
    # Function to process pose parameters
    def process_pose_params(pose_params):
        # Ensure pose_params is a NumPy array
        pose_params = np.array(pose_params).reshape(-1, 3)

        # Define the joint hierarchy in terms of parent indices based on the SMPL model
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

        # Initialize an array to hold the absolute rotations
        absolute_rotations_matrices = [np.eye(3) for _ in range(len(parents))]  # identity matrices

        # Function to compute absolute rotation
        def compute_absolute_rotation(joint_idx):
            if parents[joint_idx] == -1:
                return axis_angle_to_rotation_matrix(pose_params[joint_idx])
            else:
                parent_abs_rotation = compute_absolute_rotation(parents[joint_idx])
                return np.dot(parent_abs_rotation, axis_angle_to_rotation_matrix(pose_params[joint_idx]))

        # Calculate absolute rotations for each joint
        for i in range(len(parents)):
            absolute_rotations_matrices[i] = compute_absolute_rotation(i)

        # Convert the rotation matrices back to axis-angle representation
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

    def project_frames(interpolated_data_frames):

        # Initialize a dictionary to hold lists for each camera
        cam_lists = {
            0: [],
            1: [],
            2: [],
            3: []
        }
                    
        base_path = "/scratch_net/biwidl307_second/lgermano/behave"

        # Process interpolated frames
        for idx, input_frame in enumerate(interpolated_data_frames):

            for cam_id in [0, 1, 2, 3]:
                frame = copy.deepcopy(input_frame)
                print(f"\nProcessing frame {idx}: {frame}")  # Debug
                cam_params = load_config(cam_id, base_path, frame['date'])
                transformed_smpl_pose, transformed_smpl_trans = transform_smpl_to_camera_frame(frame['pose'], frame['trans'], cam_params)
                # print("Project debug")
                # print(transformed_smpl_pose)
                # print(transformed_smpl_trans)
                frame['pose'] = transformed_smpl_pose
                frame['trans'] = transformed_smpl_trans
                joints = render_smpl(transformed_smpl_pose, transformed_smpl_trans, frame['betas'])
                joints_numpy = [joint.cpu().numpy() for joint in joints]
                frame['joints'] = joints_numpy
                transformed_obj_pose, transformed_obj_trans =  transform_object_to_camera_frame(frame['obj_pose'], frame['obj_trans'], cam_params)
                frame['obj_pose'] = transformed_obj_pose  
                frame['obj_trans'] = transformed_obj_trans
                distances = np.asarray([np.linalg.norm(transformed_obj_trans - joint) for joint in joints_numpy])       
                frame['distances'] = distances

                cam_lists[cam_id].append(frame)

        
        scene_boundaries = []

        # Debug: Print sample from cam0_list after all operations
        if cam_lists[0]:
            print(f"\nSample from cam0_list after all operations: {cam_lists[0][0]}")

        # Gather components for normalization
        for cam_id in range(4):
            for idx in range(len(cam_lists[cam_id])):
                # Flatten the joints and add to scene_boundaries
                scene_boundaries.extend(np.array(cam_lists[cam_id][idx]['joints']).flatten())
                # Add obj_trans components to scene_boundaries
                scene_boundaries.extend(cam_lists[cam_id][idx]['obj_trans'].flatten())

        # Convert to numpy array for the min and max operations
        scene_boundaries_np = np.array(scene_boundaries)

        max_value = scene_boundaries_np.max()
        min_value = scene_boundaries_np.min()

        print(f"\nMin value for normalization: {min_value}")  # Debug
        print(f"Max value for normalization: {max_value}")    # Debug

        for cam_id in range(4):
            for idx in range(len(cam_lists[cam_id])):
                cam_lists[cam_id][idx]['norm_obj_trans'] = normalize(cam_lists[cam_id][idx]['obj_trans'], 0, 2*np.pi, min_value, max_value)
                cam_lists[cam_id][idx]['norm_joints'] = normalize(cam_lists[cam_id][idx]['joints'], 0, 2*np.pi, min_value, max_value)

        # Debug: Print sample from cam0_list after all operations
        if cam_lists[0]:
            print(f"\nSample from cam0_list after normalizing: {cam_lists[0][0]}")
            
        # Unroll angle hierarchy
        print("\nUnrolling angle hierarchy...")  # Debug
        for cam_id in range(4):
            for idx in range(len(cam_lists[cam_id])):
                cam_lists[cam_id][idx]['unrolled_pose'] = process_pose_params(cam_lists[cam_id][idx]['pose'])

        # Debug: Print sample from cam0_list after all operations
        if cam_lists[0]:
            print(f"\nSample from cam0_list after unrolling: {cam_lists[0][0]}")

        # Positional encoding
        print("\nApplying positional encoding...")  # Debug
        L = wandb.config.L
        for cam_id in range(4):
            for idx in range(len(cam_lists[cam_id])):
                cam_lists[cam_id][idx]['enc_norm_joints'] = gamma(cam_lists[cam_id][idx]['norm_joints'], L)
                cam_lists[cam_id][idx]['enc_unrolled_pose'] = gamma(cam_lists[cam_id][idx]['unrolled_pose'], L)
                cam_lists[cam_id][idx]['enc_obj_pose'] = gamma(cam_lists[cam_id][idx]['obj_pose'], L)
                cam_lists[cam_id][idx]['enc_norm_obj_trans'] = gamma(cam_lists[cam_id][idx]['norm_obj_trans'], L)

        # Debug: Print sample from cam0_list after all operations
        if cam_lists[0]:
            print(f"\nSample from cam0_list after all operations: {cam_lists[0][0]}")

        return [cam_lists[0], cam_lists[1], cam_lists[2], cam_lists[3]]

    
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
    
    class CustomCosineLR(_LRScheduler):
        def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
            self.T_max = T_max
            self.eta_min = eta_min
            super(CustomCosineLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs
            ] 
    
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
            # self.validation_losses = []

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
            # self.validation_losses = []

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

    class MLP2(pl.LightningModule):

        def __init__(self, input_dim, output_dim):
            super(MLP2, self).__init__()

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
            # self.validation_losses = []

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
        def __init__(self, cam_data):
            self.cam_data = cam_data

        def __len__(self):
            return len(self.cam_data[0])

        def __getitem__(self, idx):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Extracting the required keys and converting them to tensors
            def extract_and_convert(cam_id):
                # Keys for encoding of previous step 
                # keys = ['enc_unrolled_pose', 'enc_norm_joints', 'prev_enc_obj_pose', 'prev_enc_norm_obj_trans', 'obj_pose', 'obj_trans']
                keys = ['enc_unrolled_pose', 'enc_norm_joints', 'prev_obj_pose', 'prev_obj_trans', 'obj_pose', 'obj_trans']
                return tuple(torch.tensor(self.cam_data[cam_id][idx][key], dtype=torch.float32).to(device) for key in keys)
            #print(extract_and_convert(0), extract_and_convert(1), extract_and_convert(2), extract_and_convert(3))
            return extract_and_convert(0), extract_and_convert(1), extract_and_convert(2), extract_and_convert(3)


    class BehaveDataModule(pl.LightningDataModule):
        def __init__(self, dataset, split, batch_size=wandb.config.batch_size):
            super(BehaveDataModule, self).__init__()
            self.dataset = dataset
            self.batch_size = batch_size
            self.split = split

            self.train_indices = []
            self.test_indices = []

            train_identifiers = []
            test_identifiers = []

            for idx, data in enumerate(self.dataset.cam_data[0]):
                if data['identifier'] in self.split['train']:
                    self.train_indices.append(idx)
                    train_identifiers.append(data['identifier'])
                elif data['identifier'] in self.split['test']:
                    self.test_indices.append(idx)
                    test_identifiers.append(data['identifier'])

            # Print the list of identifiers in the train and test set
            print(f"Identifiers in train set: {train_identifiers}")
            print(f"Identifiers in test set: {test_identifiers}")

            #################
            # Using training set as validation
            # self.test_indices = self.train_indices
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

    def axis_angle_loss(pred, true):
        # Assuming pred and true are [batch_size, 3] tensors where
        # the last dimension contains the [x, y, z] coordinates of the axis-angle vector
        
        # Normalize axis vectors
        pred_axis = F.normalize(pred, dim=1)
        true_axis = F.normalize(true, dim=1)
        
        # Calculate the cosine similarity between axes
        cos_sim = F.cosine_similarity(pred_axis, true_axis, dim=1)
        
        # Calculate angle magnitudes
        pred_angle = torch.norm(pred, dim=1)
        true_angle = torch.norm(true, dim=1)
        
        # Calculate circular distance for angles
        angle_diff_options = torch.stack([
            torch.abs(pred_angle - true_angle),
            torch.abs(pred_angle - true_angle + 2 * np.pi),
            torch.abs(pred_angle - true_angle - 2 * np.pi)
        ], dim=-1)
        
        angle_diff, _ = torch.min(angle_diff_options, dim=-1)

        # Combine the two losses
        # You can weight these terms as needed
        loss = 1 - cos_sim + angle_diff

        return torch.mean(loss)

    class CombinedMLP(pl.LightningModule):
        def __init__(self, input_dim, output_stage1, input_stage2, output_stage2, input_stage3, output_dim):
            super(CombinedMLP, self).__init__()

            # Two instances of the MLP model for two stages
            self.model1 = MLP1(input_dim, output_stage1)
            self.model2 = MLP2(input_stage2, output_stage2)
            self.model3_pose = MLP3(input_stage3,output_dim)
            self.model3_trans = MLP3(input_stage3,output_dim)
            self.automatic_optimization = False
            self.validation_losses = []
            self.best_avg_loss_val = float('inf')
        
        def forward(self, cam_data):

            smpl_pose, smpl_joints, obj_pose, obj_trans, _, _ = cam_data

            x_stage1_pose = self.model1(smpl_pose)
            x_stage1_joints = self.model1(smpl_joints)

            concatenated_input = torch.cat((x_stage1_pose, x_stage1_joints), dim=1)

            x_stage2 = self.model2(concatenated_input)
            print("x_stage2 shape:", x_stage2.shape)

            # # Batched pos encoding 
            # def encode(tensor, L):
            #     # Convert to a list of tensors for each batch
            #     batch_tensors = []
            #     for batch in tensor:
            #         tensors = [gamma(p, L) for p in batch]
            #         concatenated_tensor = torch.cat(tensors, dim=0)  # change from stack to cat
            #         batch_tensors.append(concatenated_tensor)

            #     # Stack the tensors along the first dimension to create a new tensor
            #     final_tensor = torch.stack(batch_tensors)
                
            #     return final_tensor
            
            # x_stage2_enc = encode(x_stage2, L=wandb.config.L)

            # x_stage3_trans = self.model3_trans(torch.cat((x_stage2_enc,obj_trans), dim=-1))
            # x_stage3_pos = self.model3_pose(torch.cat((x_stage2_enc,obj_pose), dim=-1))

            # They should both (BATCH_SIZE,3) 

            x_stage3_trans = self.model3_trans(torch.cat((x_stage2,obj_trans), dim=-1))
            x_stage3_pos = self.model3_pose(torch.cat((x_stage2,obj_pose), dim=-1))

            return x_stage3_pos, x_stage3_trans

        def training_step(self, batch, batch_idx):
            cam0_data, cam1_data, cam2_data, cam3_data = batch

            def smooth_sign_loss(y, y_hat, alpha=wandb.config.alpha):
                # Clamp the product to prevent extreme values
                product = torch.clamp(alpha * y * y_hat, -10, 10)
                
                # Compute the stable loss
                loss = (1 - torch.sigmoid(product)).sum()
                
                return loss

            # Define weights
            lambda_1 = wandb.config.lambda_1
            lambda_2 = wandb.config.lambda_2
            lambda_3 = wandb.config.lambda_3
            lambda_4 = wandb.config.lambda_4
            
            def forward_pass_for_camera(self, cam_data):
                y_hat_stage3_pos, y_hat_stage3_trans = self(cam_data)
                y_stage3_pos = cam_data[-2]
                y_stage3_trans = cam_data[-1]
                
                return y_hat_stage3_pos, y_hat_stage3_trans, y_stage3_pos, y_stage3_trans

            # Usage example for cam0, cam1, cam2, cam3
            y_hat_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_pos_cam0, y_stage3_trans_cam0 = forward_pass_for_camera(self, cam0_data)
            y_hat_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_pos_cam1, y_stage3_trans_cam1 = forward_pass_for_camera(self, cam1_data)
            y_hat_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_pos_cam2, y_stage3_trans_cam2 = forward_pass_for_camera(self, cam2_data)
            y_hat_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_pos_cam3, y_stage3_trans_cam3 = forward_pass_for_camera(self, cam3_data)

            def geodesic_loss(r1, r2):
                """Compute the geodesic distance between two axis-angle representations."""
                return torch.min(torch.norm(r1 - r2, dim=-1), torch.norm(r1 + r2, dim=-1)).mean()

            def compute_loss(y_hat_pos, y_pos, y_hat_trans, y_trans, lambda_1, lambda_2, lambda_3, lambda_4):
                pos_loss = lambda_2 * axis_angle_loss(y_hat_pos, y_pos)
                        # lambda_1 * F.mse_loss(y_hat_pos, y_pos) + \
                        # lambda_2 * (1 - F.cosine_similarity(y_hat_pos, y_pos)) + \
                        # lambda_3 * smooth_sign_loss(y_hat_pos, y_pos) + \
                        # lambda_4 * geodesic_loss(y_hat_pos, y_pos)

                
                trans_loss = lambda_1 * F.mse_loss(y_hat_trans, y_trans)
                            # lambda_2 * (1 - F.cosine_similarity(y_hat_trans, y_trans)) + \
                            # lambda_3 * smooth_sign_loss(y_hat_trans, y_trans)
                
                return pos_loss + trans_loss

            loss_cam0 = compute_loss(y_hat_stage3_pos_cam0, y_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_trans_cam0, lambda_1, lambda_2, lambda_3,lambda_4)
            loss_cam1 = compute_loss(y_hat_stage3_pos_cam1, y_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_trans_cam1, lambda_1, lambda_2, lambda_3,lambda_4)
            loss_cam2 = compute_loss(y_hat_stage3_pos_cam2, y_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_trans_cam2, lambda_1, lambda_2, lambda_3,lambda_4)
            loss_cam3 = compute_loss(y_hat_stage3_pos_cam3, y_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_trans_cam3, lambda_1, lambda_2, lambda_3,lambda_4)
            total_loss = (loss_cam1 + loss_cam0 + loss_cam2 + loss_cam3)**2
            # print("total_loss:", total_loss)

            avg_loss = total_loss.mean()
            # print("avg_loss:", avg_loss)


            # print("Shape of loss_cam1_stage2:", loss_cam1_stage2.shape)
            # print("Shape of total_loss:", total_loss.shape)
            # print("Shape of avg_loss:", avg_loss.shape)

            # Log the individual and average losses to wandb with extended labels
            wandb.log({
                'Train Trans+Angle Batch-Averaged Average 4Cameras': avg_loss.item(),
                # "Original_Stage1_Loss": loss_original_stage1.item(),
                # "Camera1_Stage2_Loss": loss_cam1_stage2.item(),
                # Add other individual losses if needed
            })

            
            # Backward pass and optimization
            self.manual_backward(avg_loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

            return avg_loss

        def validation_step(self, batch, batch_idx):
            cam0_data, cam1_data, cam2_data, cam3_data = batch

            def forward_pass_for_camera(self, cam_data):
                y_hat_stage3_pos, y_hat_stage3_trans = self(cam_data)
                y_stage3_pos = cam_data[-2]
                y_stage3_trans = cam_data[-1]
                
                return y_hat_stage3_pos, y_hat_stage3_trans, y_stage3_pos, y_stage3_trans

            # Usage example for cam0, cam1, cam2, cam3
            y_hat_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_pos_cam0, y_stage3_trans_cam0 = forward_pass_for_camera(self, cam0_data)
            y_hat_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_pos_cam1, y_stage3_trans_cam1 = forward_pass_for_camera(self, cam1_data)
            y_hat_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_pos_cam2, y_stage3_trans_cam2 = forward_pass_for_camera(self, cam2_data)
            y_hat_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_pos_cam3, y_stage3_trans_cam3 = forward_pass_for_camera(self, cam3_data)

            def compute_loss_val(y_hat_pos, y_pos, y_hat_trans, y_trans):
                pos_loss = axis_angle_loss(y_hat_pos, y_pos)  
                trans_loss = F.mse_loss(y_hat_trans, y_trans)
                
                return pos_loss + trans_loss, pos_loss, trans_loss

            # Compute loss and component losses for each camera
            loss_cam0, pos_loss_cam0, trans_loss_cam0 = compute_loss_val(y_hat_stage3_pos_cam0, y_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_trans_cam0)
            loss_cam1, pos_loss_cam1, trans_loss_cam1 = compute_loss_val(y_hat_stage3_pos_cam1, y_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_trans_cam1)
            loss_cam2, pos_loss_cam2, trans_loss_cam2 = compute_loss_val(y_hat_stage3_pos_cam2, y_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_trans_cam2)
            loss_cam3, pos_loss_cam3, trans_loss_cam3 = compute_loss_val(y_hat_stage3_pos_cam3, y_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_trans_cam3)

            # Calculate average losses
            avg_loss = (loss_cam0 + loss_cam1 + loss_cam2 + loss_cam3) / 4
            avg_loss_pos = (pos_loss_cam0 + pos_loss_cam1 + pos_loss_cam2 + pos_loss_cam3) / 4
            avg_loss_trans = (trans_loss_cam0 + trans_loss_cam1 + trans_loss_cam2 + trans_loss_cam3) / 4

            # Log losses using wandb with extended labels
            wandb.log({
                'Val Trans+Angle Batch-Averaged Average 4Cameras': avg_loss.item(),
                'Val AxisAngleLoss Angle Batch-Averaged Average 4Cameras': avg_loss_pos.item(),
                'Val MSE Trans Batch-Averaged Average 4Cameras': avg_loss_trans.item()
            })

            self.log('val_loss', avg_loss, prog_bar=False, logger=False)

            self.validation_losses.append(avg_loss)
            return {'val_loss': avg_loss}
        
        def on_validation_epoch_end(self):

            avg_val_loss = torch.mean(torch.tensor(self.validation_losses))

            wandb.log({
                'Val Trans+Angle Epoch-Averaged Batch-Averaged Average 4Cameras': avg_val_loss.item()
                })
            self.log('avg_val_loss', avg_val_loss, prog_bar=True, logger=True)
            wandb.log({"Learning Rate": self.optimizer.param_groups[0]['lr']})
            if avg_val_loss < self.best_avg_loss_val:
                self.best_avg_loss_val = avg_val_loss
                print(f"Best number of epochs:{self.current_epoch}")
                
                # Save the model
                model_save_path = f'/scratch_net/biwidl307/lgermano/H2O/trained_models/H2O/model_{wandb.run.name}_epoch_{self.current_epoch}.pt'
                torch.save(self.state_dict(), model_save_path)
                print(f'Model saved to {model_save_path}')

            self.validation_losses = []  # reset for the next epoch
            self.lr_scheduler.step(avg_val_loss)  # Update

        def test_step(self, batch, batch_idx):
            cam0_data, cam1_data, cam2_data, cam3_data = batch
            
            def forward_pass_for_camera(self, cam_data):
                y_hat_stage3_pos, y_hat_stage3_trans = self(cam_data)
                y_stage3_pos = cam_data[-2]
                y_stage3_trans = cam_data[-1]
                
                return y_hat_stage3_pos, y_hat_stage3_trans, y_stage3_pos, y_stage3_trans

            # Usage example for cam0, cam1, cam2, cam3
            y_hat_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_pos_cam0, y_stage3_trans_cam0 = forward_pass_for_camera(self, cam0_data)
            y_hat_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_pos_cam1, y_stage3_trans_cam1 = forward_pass_for_camera(self, cam1_data)
            y_hat_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_pos_cam2, y_stage3_trans_cam2 = forward_pass_for_camera(self, cam2_data)
            y_hat_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_pos_cam3, y_stage3_trans_cam3 = forward_pass_for_camera(self, cam3_data)

            def compute_loss_val(y_hat_pos, y_pos, y_hat_trans, y_trans):
                pos_loss = F.mse_loss(y_hat_pos, y_pos)  
                trans_loss = F.mse_loss(y_hat_trans, y_trans)
                
                return pos_loss + trans_loss

            loss_cam0 = compute_loss_val(y_hat_stage3_pos_cam0, y_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_trans_cam0)
            loss_cam1 = compute_loss_val(y_hat_stage3_pos_cam1, y_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_trans_cam1)
            loss_cam2 = compute_loss_val(y_hat_stage3_pos_cam2, y_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_trans_cam2)
            loss_cam3 = compute_loss_val(y_hat_stage3_pos_cam3, y_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_trans_cam3)
            avg_loss_test = (loss_cam1 + loss_cam0 + loss_cam2 + loss_cam3)/4


            self.log('avg_loss_test', avg_loss_test, prog_bar=False, logger=False)

            wandb.log({
                'Batch-Averaged Average 4Cameras MSE Trans and Angle Test': avg_loss_test.item()
                })
            # Log the individual losses for testing to wandb
            # wandb.log({
            #     "loss_test_stage1": loss_original_stage1.item(),
            #     "loss_test_stage2": loss_cam1_stage2.item()
            # })
            return {'test_loss': avg_loss_test}

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
            'scheduler': CustomCyclicLR(optimizer, base_lr=1e-7, max_lr=5e-3, step_size=1, mode='exp_range'),
            'interval': 'step',  # step-based updates i.e. batch
            #'monitor' : 'avg_val_loss',
            'name': 'custom_clr'
            }

            # scheduler = {
            #     'scheduler': CustomCosineLR(optimizer, T_max=100, eta_min=1e-7),
            #     'interval': 'step',  # epoch-based updates
            #     'monitor' : 'avg_val_loss',
            #     'name': 'custom_cosine_lr'
            # }

            self.optimizer = optimizer  # store optimizer as class variable for logging learning rate
            self.lr_scheduler = scheduler['scheduler']  # store scheduler as class variable for updating in on_validation_epoch_end
            return [optimizer], [scheduler]
   
    #####################################################################################################################################
    # Dataset creation
    # Change .pt name when creating a new one
    data_file_path = '/scratch_net/biwidl307/lgermano/H2O/datasets/behave_test8.pkl'

    # Check if the data has already been saved
    if os.path.exists(data_file_path):
        # Load the saved data
        with open(data_file_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        #Create a dataset

        base_path = "/scratch_net/biwidl307_second/lgermano/behave"
        #labels = sorted(glob.glob(os.path.join(base_path, "sequences", "*")))
        labels = sorted(os.listdir(os.path.join(base_path, "sequences")))
        labels = labels[:4]
        dataset = []

        # Process each label separately
        for label in labels:
            #label_name = os.path.basename(label_path)
            selected_files = sorted(glob.glob(os.path.join(base_path,"sequences",label,"t*.000")))
            print(selected_files)

            print(f"Processing {label} with {len(selected_files)} frames.")
            all_data_frames = []

            # Process all frames for the current label
            for idx, frame_folder in enumerate(selected_files):
                
                frame_data = {}

                def get_frame_data(frame_folder):
                    # For object path
                    obj_pattern = os.path.join(frame_folder, "*", "fit01", "*_fit.pkl")
                    obj_matches = glob.glob(obj_pattern)
                    
                    if obj_matches:
                        frame_data['obj_path']= obj_matches[0]
                        #print(frame_data['obj_path'])
                    else:
                        obj_path = None

                    # Iterate over each match and extract the object name
                    for obj_match in obj_matches:
                        obj_name = obj_match.split(os.sep)[-3]
                        frame_data['obj_template_path'] = os.path.join(base_path, "objects", obj_name, obj_name + ".obj")
                        print(frame_data['obj_template_path'])

                    # For SMPL path
                    smpl_path = os.path.join(frame_folder, "person", "fit02", "person_fit.pkl")
                    frame_data['smpl_path'] = smpl_path
                    
                    
                    frame_data['scene'] = frame_folder.split('/')[-2]
                    frame_data['date'] = frame_folder.split('/')[-2].split('_')[0]
                    
                    return None

                get_frame_data(frame_folder)
                
                smpl_data = load_pickle(frame_data['smpl_path'])
                frame_data['pose'] = smpl_data['pose'][:72]
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

            # Interpolate and project
            interpolated_data_frames = interpolate_frames(all_data_frames)
            dataset_batch = project_frames(interpolated_data_frames)

            # Save the batch data for future use
            data_file_path = f'/scratch_net/biwidl307/lgermano/H2O/datasets/{label}.pkl'
            with open(data_file_path, 'wb') as f:
                pickle.dump(dataset_batch, f)

            print(f"Saved data for {label} to {data_file_path}")

            # Clear and collect garbage to free up memory
            all_data_frames.clear()
            del interpolated_data_frames
            del dataset_batch
            gc.collect()

        # Gather all pickled files into one single dataset

        # Define the base path
        base_path = "/scratch_net/biwidl307_second/lgermano/behave"
        dataset = []

        # Get the list of sequence labels and sort them
        labels = sorted(os.listdir(os.path.join(base_path, "sequences")))
        print(labels)

        # Filter out labels that start with "Date03" for validation set
        val_labels = [label for label in labels if label.startswith("Date03")]

        # Remaining labels for training set
        train_labels = [label for label in labels if label not in val_labels]

        # Split the training labels into two sets and add val_labels to each
        train_set_1 = val_labels + train_labels[:int(len(train_labels) / 2)]
        #print(train_set_1)
        train_set_2 = val_labels + train_labels[int(len(train_labels) / 2):]
        print(train_set_2)
        
        for label in train_set_2:
            #label_name = os.path.basename(label_path)
            print(label)
            data_file_path = f'/scratch_net/biwidl307/lgermano/H2O/datasets/{label}.pkl'

            with open(data_file_path, 'rb') as f:
                dataset_batch = pickle.load(f)

            dataset.append(dataset_batch)

        # Save the final dataset
        final_data_file_path = f'/scratch_net/biwidl307/lgermano/H2O/datasets/{wandb.run.name}_full.pkl'
        with open(final_data_file_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"\nProcessing completed. Final dataset saved in {final_data_file_path}")
  
    cam_data = {0: [], 1: [], 2: [], 3: []}  # Initialize a dictionary to hold data for each camera
    print(f"Dataset length: {len(dataset)}")

    for scene in range(len(dataset)):
        print(f"Grouping scene: {scene}")
        for cam_id in range(4):
            for idx in range(len(dataset[scene][cam_id])):
                print(dataset[scene][cam_id][idx]['obj_pose'])
                data_dict = {}  # Initialize an empty dictionary to hold data for this index
                if idx == 0:
                    # first of all sequences
                    data_dict = {
                        'enc_unrolled_pose': dataset[scene][cam_id][idx]['enc_unrolled_pose'],
                        'enc_norm_joints': dataset[scene][cam_id][idx]['enc_norm_joints'],
                        'prev_enc_obj_pose': dataset[scene][cam_id][idx]['enc_obj_pose'],
                        'prev_enc_norm_obj_trans': dataset[scene][cam_id][idx]['enc_norm_obj_trans'],
                        'prev_obj_pose': dataset[scene][cam_id][idx]['obj_pose'],
                        'prev_obj_trans':  dataset[scene][cam_id][idx]['obj_trans'],
                        'obj_pose': dataset[scene][cam_id][idx]['obj_pose'],
                        'obj_trans': dataset[scene][cam_id][idx]['obj_trans'],
                        'identifier': dataset[scene][cam_id][idx]['scene']
                    }
                    cam_data[cam_id].append(data_dict)  # Append the dictionary to the list for this camera
                else:
                    if dataset[scene][cam_id][idx]['scene'] == dataset[scene][cam_id][idx-1]['scene']:
                        data_dict = {
                            'enc_unrolled_pose': dataset[scene][cam_id][idx]['enc_unrolled_pose'],
                            'enc_norm_joints': dataset[scene][cam_id][idx]['enc_norm_joints'],
                            'prev_enc_obj_pose': dataset[scene][cam_id][idx-1]['enc_obj_pose'],
                            'prev_enc_norm_obj_trans': dataset[scene][cam_id][idx-1]['enc_norm_obj_trans'],
                            'prev_obj_pose': dataset[scene][cam_id][idx-1]['obj_pose'],
                            'prev_obj_trans':  dataset[scene][cam_id][idx-1]['obj_trans'],
                            'obj_pose': dataset[scene][cam_id][idx]['obj_pose'],
                            'obj_trans': dataset[scene][cam_id][idx]['obj_trans'],
                            'identifier': dataset[scene][cam_id][idx]['scene']
                        }
                        cam_data[cam_id].append(data_dict)  # Append the dictionary to the list for this camera
                    else:
                        # first of a sequence uses object parameters of the same frame
                        data_dict = {
                            'enc_unrolled_pose': dataset[scene][cam_id][idx]['enc_unrolled_pose'],
                            'enc_norm_joints': dataset[scene][cam_id][idx]['enc_norm_joints'],
                            'prev_enc_obj_pose': dataset[scene][cam_id][idx]['enc_obj_pose'],
                            'prev_enc_norm_obj_trans': dataset[scene][cam_id][idx]['enc_norm_obj_trans'],
                            'prev_obj_pose': dataset[scene][cam_id][idx]['obj_pose'],
                            'prev_obj_trans':  dataset[scene][cam_id][idx]['obj_trans'],
                            'obj_pose': dataset[scene][cam_id][idx]['obj_pose'],
                            'obj_trans': dataset[scene][cam_id][idx]['obj_trans'],
                            'identifier': dataset[scene][cam_id][idx]['scene']
                        }            
                        cam_data[cam_id].append(data_dict)  # Append the dictionary to the list for this camera

    # for scene in range(len(dataset)):
    # print(f"Grouping scene: {scene}")
    # for cam_id in range(4):
    #     for idx in range(len(dataset[cam_id])):
    #         data_dict = {}  # Initialize an empty dictionary to hold data for this index
    #         if idx == 0:
    #             # first of all sequences
    #             data_dict = {
    #                 'enc_unrolled_pose': dataset[cam_id][idx]['enc_unrolled_pose'],
    #                 'enc_norm_joints': dataset[cam_id][idx]['enc_norm_joints'],
    #                 'prev_enc_obj_pose': dataset[cam_id][idx]['enc_obj_pose'],
    #                 'prev_enc_norm_obj_trans': dataset[cam_id][idx]['enc_norm_obj_trans'],
    #                 'prev_obj_pose': dataset[cam_id][idx]['obj_pose'],
    #                 'prev_obj_trans':  dataset[cam_id][idx]['obj_trans'],
    #                 'obj_pose': dataset[cam_id][idx]['obj_pose'],
    #                 'obj_trans': dataset[cam_id][idx]['obj_trans'],
    #                 'identifier': dataset[cam_id][idx]['scene']
    #             }
    #             cam_data[cam_id].append(data_dict)  # Append the dictionary to the list for this camera
    #         else:
    #             if dataset[cam_id][idx]['scene'] == dataset[cam_id][idx-1]['scene']:
    #                 data_dict = {
    #                     'enc_unrolled_pose': dataset[cam_id][idx]['enc_unrolled_pose'],
    #                     'enc_norm_joints': dataset[cam_id][idx]['enc_norm_joints'],
    #                     'prev_enc_obj_pose': dataset[cam_id][idx-1]['enc_obj_pose'],
    #                     'prev_enc_norm_obj_trans': dataset[cam_id][idx-1]['enc_norm_obj_trans'],
    #                     'prev_obj_pose': dataset[cam_id][idx-1]['obj_pose'],
    #                     'prev_obj_trans':  dataset[cam_id][idx-1]['obj_trans'],
    #                     'obj_pose': dataset[cam_id][idx]['obj_pose'],
    #                     'obj_trans': dataset[cam_id][idx]['obj_trans'],
    #                     'identifier': dataset[cam_id][idx]['scene']
    #                 }
    #                 cam_data[cam_id].append(data_dict)  # Append the dictionary to the list for this camera
    #             else:
    #                 # first of a sequence uses object parameters of the same frame
    #                 data_dict = {
    #                     'enc_unrolled_pose': dataset[cam_id][idx]['enc_unrolled_pose'],
    #                     'enc_norm_joints': dataset[cam_id][idx]['enc_norm_joints'],
    #                     'prev_enc_obj_pose': dataset[cam_id][idx]['enc_obj_pose'],
    #                     'prev_enc_norm_obj_trans': dataset[cam_id][idx]['enc_norm_obj_trans'],
    #                     'prev_obj_pose': dataset[cam_id][idx]['obj_pose'],
    #                     'prev_obj_trans':  dataset[cam_id][idx]['obj_trans'],
    #                     'obj_pose': dataset[cam_id][idx]['obj_pose'],
    #                     'obj_trans': dataset[cam_id][idx]['obj_trans'],
    #                     'identifier': dataset[cam_id][idx]['scene']
    #                 }            
    #                 cam_data[cam_id].append(data_dict)  # Append the dictionary to the list for this camera

    ########################################################################################################################
    # Data module creation
    # Change .pt name when creating a new one     
    # print("Debug")
    # print(dataset[0][0]['pose'])
    # print(dataset[1][0]['pose'])
    # print(dataset[2][0]['pose'])
    # print(dataset[3][0]['pose'])

    dataset_grouped = BehaveDataset(cam_data)

    print(f"\nLength after grouping: {len(cam_data[0])}")
 
    path_to_file = "/scratch_net/biwidl307_second/lgermano/behave/split.json"
    split_dict = load_split_from_path(path_to_file)

    # Train and validate your model with the current set of hyperparameters
    data_module = BehaveDataModule(dataset_grouped, split=split_dict, batch_size=BATCH_SIZE)

    # Combine wandb.run.name to create a unique name for the saved file
    save_file_name = f"{wandb.run.name}.pt"

    # Define the local path where the data module will be saved
    data_file_path = '/scratch_net/biwidl307/lgermano/H2O/data_module'
    full_save_path = os.path.join(data_file_path, save_file_name)

    # Save the data module locally
    torch.save(data_module, full_save_path)
    #########################################################################################################################
    # Train

    #Load any data module
    #save_file_name = "earthy-star-2339_noenconprev.pt"

    # Define the local path where the data module will be saved
    data_file_path = '/scratch_net/biwidl307/lgermano/H2O/data_module'
    full_save_path = os.path.join(data_file_path, save_file_name)
    
    # Load the data module back to a variable named data_module
    data_module = torch.load(full_save_path)

    # # Load a subset of indices
    # subsampled_train_indices = data_module.train_indices[::6]
    # subsampled_val_indices = data_module.test_indices[::6]  # Assuming test set is used as validation

    # # Update the original data_module with the subsampled indices
    # data_module.train_indices = subsampled_train_indices
    # data_module.test_indices = subsampled_val_indices

    # Get the sizes of the subsampled datasets
    train_size = len(data_module.train_indices)
    val_size = len(data_module.test_indices)
    test_size = len(data_module.test_indices)  # Assuming test set is used as validation

    print(f"Size of train set: {train_size}")
    print(f"Size of val set: {val_size}")
    print(f"Size of test set: {test_size}")

    # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on: {device}")

    # Initialize model
    input_dim = 72 * wandb.config.L * 2
    output_stage1 = 256
    input_stage2 = 512
    output_stage2 = 3
    input_stage3 = 6 #2 * (3 * wandb.config.L * 2) # 6
    output_dim = 3

    print(f"Wandb run name:{wandb.run.name}")
    model_combined = CombinedMLP(input_dim, output_stage1, input_stage2, output_stage2, input_stage3, output_dim)

    # Move the model to device
    model_combined.to(device)

    #Specify the path to the checkpoint
    model_path = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/H2O/model_mischievous-sorcery-2421_epoch_13.pt"

    #Load the state dict from the checkpoint into the model
    checkpoint = torch.load(model_path, map_location=device)
    model_combined.load_state_dict(checkpoint)

    # # Freeze the weights of the trans head 
    # for param in model_combined.model3_trans.parameters():
    #     param.requires_grad = False

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
            "lambda_3" : LAMBDA_3,
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