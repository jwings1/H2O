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
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset

EPOCHS = 100
best_val_loss = float("inf")
best_params = None

# Set the WANDB_CACHE_DIR environment variable
os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"

# Random Search
# # Define expanded ranges for your hyperparameters
# learning_rate_range = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
# batch_size_range = [8, 16, 32, 64, 128, 256]
# dropout_rate_range = [0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
# layer_sizes_range = [
#     [128, 128, 64, 64, 32],
#     [256, 256, 128, 128, 64],
#     [512, 512, 256, 256, 128],
#     [1024, 1024, 512, 512, 256],
#     [256, 256, 256, 128, 64, 32],
#     [512, 512, 512, 256, 128, 64],
#     [1024, 1024, 1024, 512, 256, 128],
#     [128, 128, 128, 64, 64, 32, 32],
#     [128, 128, 64, 64, 32],
#     [4096, 4096, 4096, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 512, 256, 128, 64],
#     [8192, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
#     [8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
#     [8192, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
#     [8192, 8192, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64, 32],
#     [8192, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 512, 256, 128, 64],
# ]

learning_rate_range = [1e-3]
#learning_rate_range = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
batch_size_range = [16]
dropout_rate_range = [0.05]
layer_sizes_range = [
    # [128, 128, 64, 64, 32],
    # [256, 256, 128, 128, 64],
    [512, 512, 256, 256, 128],
    # [1024, 1024, 512, 512, 256],
    # [256, 256, 256, 128, 64, 32],
    # [512, 512, 512, 256, 128, 64],
    # [1024, 1024, 1024, 512, 256, 128],
    # [128, 128, 128, 64, 64, 32, 32]
]

# Grid search over all combinations
for lr, bs, dr, layers in itertools.product(
    learning_rate_range, batch_size_range, dropout_rate_range, layer_sizes_range
):
    LEARNING_RATE = lr
    BATCH_SIZE = bs
    DROPOUT_RATE = dr
    LAYER_SIZES = layers
    # Initialize the input to 24 joints
    INITIAL_OBJ_PRED = torch.rand((BATCH_SIZE, 24))


    #trainer = Trainer(log_every_n_steps=BATCH_SIZE)  # Log every n steps

    # Initialize wandb with hyperparameters
    wandb.init(
        project="MLP",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "MLP",
            "dataset": "BEHAVE",
            "epochs": EPOCHS,  # keeping epochs constant across trials for simplicity
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "layer_sizes": LAYER_SIZES,
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

            for i in range(1, 3):
                interpolated_frame = frame1.copy()
                t = i / 3.0  # Assuming you want to interpolate at 1/3 and 2/3 positions between frame1 and frame2
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
    
        print("Start of render_smpl function.")
        
        batch_size = 1
        print(f"batch_size: {batch_size}")

        # Create the SMPL layer
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender='male',
            model_root='/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch/native/models/')
        print("SMPL_Layer created.")

        # Process pose parameters
        pose_params_start = torch.tensor(transformed_pose[:3], dtype=torch.float32)
        pose_params_rest = torch.tensor(transformed_pose[3:72], dtype=torch.float32)
        pose_params_rest[-6:] = 0
        pose_params = torch.cat([pose_params_start, pose_params_rest]).unsqueeze(0).repeat(batch_size, 1)
        print(f"pose_params shape: {pose_params.shape}")

        shape_params = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        print(f"shape_params shape: {shape_params.shape}")

        obj_trans = torch.tensor(transformed_trans, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        print(f"obj_trans shape: {obj_trans.shape}")

        # GPU mode
        cuda = torch.cuda.is_available()
        print(f"CUDA available: {cuda}")
        device = torch.device("cuda:0" if cuda else "cpu")
        print(f"Device: {device}")
        
        pose_params = pose_params.to(device)
        shape_params = shape_params.to(device)
        obj_trans = obj_trans.to(device)
        smpl_layer = smpl_layer.to(device)
        print("All tensors and models moved to device.")

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
        #                 left_foot, right_foot, head, left_shoulder, right_shoulder, left_hand, right_hand]      
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
                print(f"\nProcessing for camera {cam_id}...")
                
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

                    print(f"Distances in cam {cam_id}: {distances}.")

                    ########################################
                    # # POSE + OBJ TRANS --> DISTANCES
                    # data = {}
                    # data['angle'] = obj_pose
                    # data['trans'] = obj_trans                 
                    # transformed_smpl[-3:] =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                    # reprojected_smpl_cam1_list.append(transformed_smpl)

                    #########################################
                else:
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
                    else:
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

        # distance regressor - USING only to occupy first 24 places
        model_dist_path = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_offset_trans_smpl_augmented_resilient-wind-1110.pt"
        model_dist = torch.load(model_dist_path)
        # Set the model to evaluation mode
        model_dist.eval()

        # Process interpolated frames
        for idx, frame_data in enumerate(interpolated_data_frames):

            pose = frame_data['pose'][:72]
            trans = frame_data['trans']
            betas = frame_data['betas']
            obj_pose = frame_data['obj_pose']
            obj_trans = frame_data['obj_trans']
            #scene_name = frame_data['scene'] 
                            
            for cam_id in [1, 0, 2, 3]:
                print(f"\nProcessing for camera {cam_id}...")
                
                camera1_params = load_config(1, base_path, 'Date07')
                cam_params = load_config(cam_id, base_path, 'Date07')
                transformed_smpl = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params)
                        
                if cam_id == 1:
                    
                    reprojected_obj_cam1_list.append(obj_trans)

                    selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                    input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                    input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                    # Use the neural network model to predict the offset in the object's pose
                    candidate_distances = model_dist(input_tensor)
                    #print(candidate_distances)
                    distances = candidate_distances.detach().numpy().flatten() 
                    #print(selected_joints)
                    transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                    #print(transformed_smpl)
                    reprojected_smpl_cam1_list.append(transformed_smpl)

                    print(f"Distances in cam {cam_id}: {distances}.")

                else:
                    if cam_id == 0:
                        data = {}
                        data['angle'] = obj_pose
                        data['trans'] = obj_trans                 
                        obj_trans =  transform_object_to_camera_frame(data, camera1_params, cam_params)
                        reprojected_obj_cam0_list.append(obj_trans)

                        selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                        selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                        input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                        input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                        # Use the neural network model to predict the offset in the object's pose
                        candidate_distances = model_dist(input_tensor)
                        #print(candidate_distances)
                        distances = candidate_distances.detach().numpy().flatten() 
                        
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

                        input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                        input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                        # Use the neural network model to predict the offset in the object's pose
                        candidate_distances = model_dist(input_tensor)
                        #print(candidate_distances)
                        distances = candidate_distances.detach().numpy().flatten() 
                        
                        transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                        reprojected_smpl_cam2_list.append(transformed_smpl)
                    else:
                        data = {}
                        data['angle'] = obj_pose
                        data['trans'] = obj_trans                 
                        obj_trans =  transform_object_to_camera_frame(data, camera1_params, cam_params)

                        reprojected_obj_cam3_list.append(obj_trans)

                        selected_joints = render_smpl(transformed_smpl[:72], transformed_smpl[-3:], betas)
                        selected_joints = [joint.cpu().numpy() for joint in selected_joints]

                        input_tensor = torch.tensor(np.concatenate([transformed_smpl[:72], transformed_smpl[-3:]]), dtype=torch.float32)
                        input_tensor = input_tensor.unsqueeze(0)  # Make the tensor two-dimensional
                        # Use the neural network model to predict the offset in the object's pose
                        candidate_distances = model_dist(input_tensor)
                        #print(candidate_distances)
                        distances = candidate_distances.detach().numpy().flatten() 
                        
                        transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                        reprojected_smpl_cam3_list.append(transformed_smpl)
                
            #identifier = filename.split('/')[6]
            identifier = "Date07_Sub04_yogaball_play"
            identifiers.append(identifier)

        return reprojected_smpl_cam1_list, reprojected_smpl_cam0_list, reprojected_smpl_cam2_list, reprojected_smpl_cam3_list, reprojected_obj_cam1_list, \
                reprojected_obj_cam0_list, reprojected_obj_cam2_list, reprojected_obj_cam3_list, identifiers

    class BehaveDataset(Dataset):
        def __init__(self, src_data, tgt_data, identifiers):
            self.src_data = src_data
            self.tgt_data = tgt_data
            self.identifiers = identifiers

        def __len__(self):
            return len(self.src_data)

        def __getitem__(self, idx):
            return self.src_data[idx], self.tgt_data[idx]

    class TransformerEncoder(nn.Module):
        def __init__(self, d_model, nhead, num_layers, dim_feedforward):
            super(TransformerEncoder, self).__init__()
            self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
            
        def forward(self, src):
            return self.transformer_encoder(src)

    class TransformerDecoder(nn.Module):
        def __init__(self, d_model, nhead, num_layers, dim_feedforward, output_dim):
            super(TransformerDecoder, self).__init__()
            self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
            self.fc_out = nn.Linear(d_model, output_dim)
            
        def forward(self, tgt, memory):
            output = self.transformer_decoder(tgt, memory)
            return self.fc_out(output)

    class TransformerModel(pl.LightningModule):
        def __init__(self, d_model, nhead, num_layers, dim_feedforward, input_dim, output_dim):
            super(TransformerModel, self).__init__()
            self.encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward)
            self.decoder = TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, output_dim)
            self.fc_in = nn.Linear(input_dim, d_model)
            
        def forward(self, src, tgt):
            src = self.fc_in(src)
            tgt = self.fc_in(tgt)
            memory = self.encoder(src)
            output = self.decoder(tgt, memory)
            return output

        def training_step(self, batch, batch_idx):
            x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, x_cam0_stage2, x_cam1_stage2, x_cam2_stage2, x_cam3_stage2, y_stage2, y_cam0_stage2, y_cam2_stage2, y_cam3_stage2, _ = batch

            # Forward pass
            y_hat_stage1, y_hat_stage2 = self(x, x_cam0_stage2)
            y_hat_cam0_stage1, y_hat_cam0_stage2 = self(x_cam0, x_cam1_stage2)
            y_hat_cam2_stage1, y_hat_cam2_stage2 = self(x_cam2, x_cam2_stage2)
            y_hat_cam3_stage1, y_hat_cam3_stage2 = self(x_cam3, x_cam3_stage2)

            # Compute the losses for the first stage
            loss_original_stage1 = F.mse_loss(y_hat_stage1, y)
            loss_cam0_stage1 = F.mse_loss(y_hat_cam0_stage1, y_cam0)
            loss_cam2_stage1 = F.mse_loss(y_hat_cam2_stage1, y_cam2)
            loss_cam3_stage1 = F.mse_loss(y_hat_cam3_stage1, y_cam3)
            
            # Compute the losses for the second stage
            loss_original_stage2 = F.mse_loss(y_hat_stage2, y_stage2)
            loss_cam0_stage2 = F.mse_loss(y_hat_cam0_stage2, y_cam0_stage2)
            loss_cam2_stage2 = F.mse_loss(y_hat_cam2_stage2, y_cam2_stage2)
            loss_cam3_stage2 = F.mse_loss(y_hat_cam3_stage2, y_cam3_stage2)
            
            # # Sum the losses
            # total_loss = (loss_original_stage1 + loss_cam0_stage1 + loss_cam2_stage1 + loss_cam3_stage1 +
            #             loss_original_stage2 + loss_cam0_stage2 + loss_cam2_stage2 + loss_cam3_stage2)

            # # Average the losses
            # avg_loss = total_loss / 8

            # Sum the losses
            total_loss = (loss_original_stage2 + loss_cam0_stage2 + loss_cam2_stage2 + loss_cam3_stage2)

            # Average the losses
            avg_loss = total_loss / 4

            # Log the individual and average losses to wandb
            wandb.log({
                "loss_train": avg_loss.item(),
                "loss_original_stage1": loss_original_stage1.item(),
                "loss_original_stage2": loss_original_stage2.item(),
                # Add other individual losses if needed
            })

            self.validation_losses.append(avg_loss)
            
            # Backward pass and optimization
            self.manual_backward(avg_loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()

            return avg_loss

        def validation_step(self, batch, batch_idx):
            x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, y_stage2, y_cam0_stage2, y_cam2_stage2, y_cam3_stage2, _ = batch
            y_hat_stage1, y_hat_stage2 = self(x)
            
            # Compute validation losses for both stages
            loss_original_stage1 = F.mse_loss(y_hat_stage1, y)
            loss_original_stage2 = F.mse_loss(y_hat_stage2, y_stage2)
            avg_loss = (loss_original_stage1 + loss_original_stage2) / 2

            self.log('val_loss', avg_loss, prog_bar=True, logger=True)

            self.validation_losses.append(avg_loss)
            return {'val_loss': avg_loss}

        def on_validation_epoch_end(self):
            avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
            self.log('loss_val', avg_val_loss, prog_bar=True, logger=True)
            wandb.log({"avg_loss_val": avg_val_loss.item()})#, step=self.current_epoch)
            #self.log_scheduler_info(avg_val_loss.item())
            self.validation_losses = []  # reset for the next epoch

        def test_step(self, batch, batch_idx):
            x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, y_stage2, y_cam0_stage2, y_cam2_stage2, y_cam3_stage2, _ = batch
            y_hat_stage1, y_hat_stage2 = self(x)
            
            # Compute test losses for both stages
            loss_original_stage1 = F.mse_loss(y_hat_stage1, y)
            loss_original_stage2 = F.mse_loss(y_hat_stage2, y_stage2)
            avg_loss = (loss_original_stage1 + loss_original_stage2) / 2

            wandb.log({"loss_test": avg_loss.item()})
            # Log the individual losses for testing to wandb
            wandb.log({
                "loss_test_stage1": loss_original_stage1.item(),
                "loss_test_stage2": loss_original_stage2.item()
            })
            return avg_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler = {
                'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.5),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
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
    
        print(f"Detected {len(selected_files)} frames.")

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

    # Prepare data for the Transformer
    src_data = [torch.cat([reprojected_smpl_cam1_dist[i], 
                        reprojected_smpl_cam0_dist[i], 
                        reprojected_smpl_cam2_dist[i], 
                        reprojected_smpl_cam3_dist[i]]) for i in range(len(reprojected_smpl_cam1_dist))]

    tgt_data = [torch.cat([reprojected_smpl_cam1_tri[i], 
                        reprojected_smpl_cam0_tri[i], 
                        reprojected_smpl_cam2_tri[i], 
                        reprojected_smpl_cam3_tri[i]]) for i in range(len(reprojected_smpl_cam1_tri))]

    # Create the dataset
    dataset = BehaveDataset(src_data, tgt_data, identifiers_dist)

    # Initialize the Transformer model
    d_model = 300  
    nhead = 4  
    num_layers = 2  
    dim_feedforward = 2048  
    input_dim = 99  # 75 + 24
    output_dim = 3  # Position of the object

    model = TransformerModel(d_model, nhead, num_layers, dim_feedforward, input_dim, output_dim)
    trainer = pl.Trainer(max_epochs=wandb.config.epochs)
    trainer.fit(model, datamodule=data_module)

    # Adjusted computation for average validation loss
    # TO BE FIXED
    if combined_model.validation_losses:
        avg_val_loss = torch.mean(torch.stack(combined_model.validation_losses)).item()
    else:
        avg_val_loss = float('inf')

    # If current validation loss is the best, update best loss and best params
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_params = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "layer_sizes": LAYER_SIZES
        }

    # Optionally, to test the model:
    trainer.test(combined_model, datamodule=data_module)

    # Save the model using WandB run ID
    filename = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_offset_trans_smpl_augmented_{wandb.run.name}.pt"

    # Save the model
    torch.save(combined_model, filename)

    # Finish the current W&B run
    wandb.finish()


# After all trials, print the best set of hyperparameters
print("Best Validation Loss:", best_val_loss)
print("Best Hyperparameters:", best_params)