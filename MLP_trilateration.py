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

EPOCHS = 500
best_val_loss = float("inf")
best_params = None
N_MLPS = 3

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

#learning_rate_range = [5e-4]
learning_rate_range = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
batch_size_range = [64]
dropout_rate_range = [0.3]
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

    def project_frames(interpolated_data_frames):
        
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
        # distance regressor
        model_path = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_offset_trans_smpl_augmented_resilient-wind-1110.pt"
        model = torch.load(model_path)
        # Set the model to evaluation mode
        model.eval()


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
                    candidate_distances = model(input_tensor)
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
                        candidate_distances = model(input_tensor)
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
                        candidate_distances = model(input_tensor)
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
                        candidate_distances = model(input_tensor)
                        #print(candidate_distances)
                        distances = candidate_distances.detach().numpy().flatten() 
                        
                        transformed_smpl = np.concatenate([np.concatenate(selected_joints), distances])
                        reprojected_smpl_cam3_list.append(transformed_smpl)
            
        #identifier = filename.split('/')[6]
        identifier = "Date07_Sub04_yogaball_play"
        identifiers.append(identifier)

        return reprojected_smpl_cam1_list, reprojected_smpl_cam0_list, reprojected_smpl_cam2_list, reprojected_smpl_cam3_list, reprojected_obj_cam1_list, \
                reprojected_obj_cam0_list, reprojected_obj_cam2_list, reprojected_obj_cam3_list, identifiers

    # Update to the BehaveDataset class
    class BehaveDataset(Dataset):
        def __init__(self, smpl_inputs, smpl_reprojected_cam0, smpl_reprojected_cam2, smpl_reprojected_cam3, 
                    obj_labels, obj_reprojected_cam0, obj_reprojected_cam2, obj_reprojected_cam3, 
                    identifiers):
            
            self.inputs = smpl_inputs
            self.reprojected_cam0_inputs = smpl_reprojected_cam0
            self.reprojected_cam2_inputs = smpl_reprojected_cam2
            self.reprojected_cam3_inputs = smpl_reprojected_cam3
            
            self.labels = obj_labels
            self.reprojected_cam0_labels = obj_reprojected_cam0
            self.reprojected_cam2_labels = obj_reprojected_cam2
            self.reprojected_cam3_labels = obj_reprojected_cam3
            
            self.identifiers = identifiers

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.inputs[idx], dtype=torch.float32),
                torch.tensor(self.reprojected_cam0_inputs[idx], dtype=torch.float32),
                torch.tensor(self.reprojected_cam2_inputs[idx], dtype=torch.float32),
                torch.tensor(self.reprojected_cam3_inputs[idx], dtype=torch.float32),
                
                torch.tensor(self.labels[idx], dtype=torch.float32),
                torch.tensor(self.reprojected_cam0_labels[idx], dtype=torch.float32),
                torch.tensor(self.reprojected_cam2_labels[idx], dtype=torch.float32),
                torch.tensor(self.reprojected_cam3_labels[idx], dtype=torch.float32),
                
                self.identifiers[idx]
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

        def train_dataloader(self):
            train_dataset = Subset(self.dataset, self.train_indices)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            # Assuming validation set is not provided, so using test set as validation
            test_dataset = Subset(self.dataset, self.test_indices)
            return DataLoader(test_dataset, batch_size=self.batch_size)

        def test_dataloader(self):
            test_dataset = Subset(self.dataset, self.test_indices)
            return DataLoader(test_dataset, batch_size=self.batch_size)

            # 3. MLP Model
    class MLP(pl.LightningModule):

        def __init__(self, input_dim, output_dim):
            super(MLP, self).__init__()

            self.automatic_optimization = False

            # Use layer_sizes from wandb.config to create the architecture
            layer_sizes = [input_dim] + wandb.config.layer_sizes + [output_dim]
            self.linears = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
            
            # Batch normalization layers based on the layer sizes
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(size) for size in wandb.config.layer_sizes])

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
                if i < len(self.bns):
                    x = self.bns[i](x)  # Batch normalization before activation
                x = F.relu(x)  # Activation function
                x = self.dropout(x)
                
            x = self.linears[-1](x)
            return x

        def training_step(self, batch, batch_idx):
            x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
            
            # Compute the predictions
            y_hat = self(x)
            y_hat_cam0 = self(x_cam0)
            y_hat_cam2 = self(x_cam2)
            y_hat_cam3 = self(x_cam3)
            
            # # Compute the losses using geodesic distance
            # loss_original = geodesic_loss(y_hat, y)
            # loss_cam0 = geodesic_loss(y_hat_cam0, y_cam0)
            # loss_cam2 = geodesic_loss(y_hat_cam2, y_cam2)
            # loss_cam3 = geodesic_loss(y_hat_cam3, y_cam3)


            # Compute the losses using Mean Squared Error (MSE) - trans
            loss_original = F.mse_loss(y_hat, y)
            loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
            loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
            loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
            
            # Average the losses
            avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
            
            # Log the average loss
            wandb.log({"loss_train": avg_loss.item()})#, step=self.current_epoch)       
            self.manual_backward(avg_loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()
            return avg_loss

        def validation_step(self, batch, batch_idx):
            x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
            y_hat = self(x)
            loss_original = F.mse_loss(y_hat, y)
            y_hat_cam0 = self(x_cam0)
            loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
            y_hat_cam2 = self(x_cam2)
            loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
            y_hat_cam3 = self(x_cam3)
            loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
            avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
            self.validation_losses.append(avg_loss)
            wandb.log({"loss_val": avg_loss.item()})#, step=self.current_epoch)
            return {'val_loss': avg_loss}

        def on_validation_epoch_end(self):
            avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
            self.log('loss_val', avg_val_loss, prog_bar=True, logger=True)
            wandb.log({"avg_loss_val": avg_val_loss.item()})#, step=self.current_epoch)
            self.log_scheduler_info(avg_val_loss.item())
            self.validation_losses = []  # reset for the next epoch

        def log_scheduler_info(self, val_loss):
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler = scheduler[0]
            
            # Log learning rate of the optimizer
            for idx, param_group in enumerate(self.optimizers().param_groups):
                wandb.log({f"learning_rate_{idx}": param_group['lr']})
            
            # Log best metric value seen so far by the scheduler
            best_metric_val = scheduler.best
            wandb.log({"best_val_loss": best_metric_val})

            # Log number of epochs since last improvements
            epochs_since_improvement = scheduler.num_bad_epochs
            wandb.log({"epochs_since_improvement": epochs_since_improvement})
            
            # Manually step the scheduler
            scheduler.step(val_loss)
            
        def test_step(self, batch, batch_idx):
            x, x_cam0, x_cam2, x_cam3, y, y_cam0, y_cam2, y_cam3, _ = batch
            y_hat = self(x)
            loss_original = F.mse_loss(y_hat, y)
            y_hat_cam0 = self(x_cam0)
            loss_cam0 = F.mse_loss(y_hat_cam0, y_cam0)
            y_hat_cam2 = self(x_cam2)
            loss_cam2 = F.mse_loss(y_hat_cam2, y_cam2)
            y_hat_cam3 = self(x_cam3)
            loss_cam3 = F.mse_loss(y_hat_cam3, y_cam3)
            avg_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4
            wandb.log({"loss_test": avg_loss.item()})
            return avg_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler = {
                'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.5),
                'monitor': 'loss_val',
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]

    # 4. Training using PyTorch Lightnings
    # Integrating the loading and dataset creation
    behave_seq = "/scratch_net/biwidl307_second/lgermano/behave/sequences/Date07_Sub04_yogaball_play"
    base_path = "/scratch_net/biwidl307_second/lgermano/behave"

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
    print("hello")
    print(len(interpolated_data_frames))
    reprojected_smpl_cam1_list, reprojected_smpl_cam0_list, reprojected_smpl_cam2_list, reprojected_smpl_cam3_list, \
    reprojected_obj_cam1_list, reprojected_obj_cam0_list, reprojected_obj_cam2_list, reprojected_obj_cam3_list, identifiers = project_frames(interpolated_data_frames)
    
    input_dim = 96
    output_dim = 3

    dataset = BehaveDataset(
        reprojected_smpl_cam1_list,
        reprojected_smpl_cam0_list,
        reprojected_smpl_cam2_list,
        reprojected_smpl_cam3_list,
        reprojected_obj_cam1_list,
        reprojected_obj_cam0_list,
        reprojected_obj_cam2_list,
        reprojected_obj_cam3_list,
        identifiers
    )

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

    model_tri = MLP(input_dim, output_dim)
    trainer = pl.Trainer(max_epochs=wandb.config.epochs)
    trainer.fit(model_tri, datamodule=data_module)

    # Adjusted computation for average validation loss
    # TO BE FIXED
    if model_tri.validation_losses:
        avg_val_loss = torch.mean(torch.stack(model_tri.validation_losses)).item()
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
    trainer.test(model_tri, datamodule=data_module)

    # Save the model using WandB run ID
    filename = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_offset_trans_smpl_augmented_{wandb.run.name}.pt"

    # Save the model
    torch.save(model_tri, filename)

    # Finish the current W&B run
    wandb.finish()


# After all trials, print the best set of hyperparameters
print("Best Validation Loss:", best_val_loss)
print("Best Hyperparameters:", best_params)
