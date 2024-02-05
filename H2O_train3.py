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

# Function to create timestamp
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_parser():
    parser = argparse.ArgumentParser(description='Training script for H20 model.')
    parser.add_argument('--first_option', choices=['pose','pose_trace', 'unrolled_pose', 'unrolled_pose_trace', 'enc_unrolled_pose', 'enc_unrolled_pose_trace'], help='Specify the first option.')
    parser.add_argument('--second_option', choices=['joints', 'distances', 'joints_trace','norm_joints', 'norm_joints_trace', 'enc_norm_joints',   'enc_norm_joints_trace'], help='Specify the second option.')
    parser.add_argument('--third_option', choices=['obj_pose', 'enc_obj_pose'], help='Specify the third option.')
    parser.add_argument('--fourth_option', choices=['obj_trans', 'norm_obj_trans', 'enc_norm_obj_trans'], help='Specify the fourth option.')
    parser.add_argument('--scene', default=['scene'],help='Include scene in the options.')
    parser.add_argument('--learning_rate', nargs='+', type=float, default=[1e-3])
    parser.add_argument('--epochs', nargs='+', type=int, default=[50])
    parser.add_argument('--batch_size', nargs='+', type=int, default=[32])
    parser.add_argument('--dropout_rate', nargs='+', type=float, default=[0.00])
    parser.add_argument('--alpha', nargs='+', type=float, default=[1])
    parser.add_argument('--lambda_1', nargs='+', type=float, default=[1], help='Weight for mse_loss.')
    parser.add_argument('--lambda_2', nargs='+', type=float, default=[1], help='Weight for cosine_similarity.')
    parser.add_argument('--lambda_3', nargs='+', type=float, default=[1], help='Weight for custom smooth_sign_loss.')
    parser.add_argument('--lambda_4', nargs='+', type=float, default=[1], help='Weight for geodesic_loss.')
    parser.add_argument('--L', nargs='+', type=int, default=[4], choices=[4])
    parser.add_argument('--optimizer', nargs='+', default=["AdamW"], choices=["AdamW", "Adagrad", "Adadelta", "LBFGS", "Adam", "RMSprop"])
    parser.add_argument('--layer_sizes_1', nargs='+', type=int, default=[[256, 256, 256]])
    parser.add_argument('--layer_sizes_3', nargs='+', type=int, default=[[64, 128, 256, 128, 64]])
    parser.add_argument('--name', default=timestamp())

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    print(args)

    best_overall_avg_loss_val = float('inf')
    best_params = None

    # Set the WANDB_CACHE_DIR environment variable
    os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"

    # Assigning the parsed values to corresponding variables
    learning_rate_range = args.learning_rate
    epochs_range = args.epochs
    batch_size_range = args.batch_size
    dropout_rate_range = args.dropout_rate
    alpha_range = args.alpha
    lambda_1_range = args.lambda_1
    lambda_2_range = args.lambda_2
    lambda_3_range = args.lambda_3
    lambda_4_range = args.lambda_4
    L_range = args.L
    optimizer_list = args.optimizer
    layer_sizes_range_1 = args.layer_sizes_1
    layer_sizes_range_3 = args.layer_sizes_3
    SMPL_pose = args.first_option
    SMPL_joints = args.second_option
    OBJ_pose = args.third_option
    OBJ_trans = args.fourth_option
    scene = args.scene
    name = args.name

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

        def load_intrinsics_and_distortion(camera_id, base_path):
            calib_path = os.path.join(base_path, 'calibs', 'intrinsics', str(camera_id), 'calibration.json')
            with open(calib_path, 'r') as f:
                calib_data = json.load(f)
                color_intrinsics = calib_data['color']
                return {
                    'fx': color_intrinsics['fx'],
                    'fy': color_intrinsics['fy'],
                    'cx': color_intrinsics['cx'],
                    'cy': color_intrinsics['cy']
                }, {
                    'k1': color_intrinsics['k1'],
                    'k2': color_intrinsics['k2'],
                    'k3': color_intrinsics['k3'],
                    'p1': color_intrinsics['p1'],
                    'p2': color_intrinsics['p2']
                }

        def plot_obj_in_camera_frame(obj_pose, obj_trans, obj_template_path):
            # Load obj template
            #object_template = "/scratch_net/biwidl307_second/lgermano/behave/objects/stool/stool.obj"
            object_mesh = o3d.io.read_triangle_mesh(obj_template_path)
            object_vertices = np.asarray(object_mesh.vertices)
            
            # Debug: #print object vertices before any transformation
            ##print("Object vertices before any transformation: ", object_vertices)

            # Compute the centroid of the object
            centroid = np.mean(object_vertices, axis=0)

            # Translate all vertices such that the object's centroid is at the origin
            object_vertices = object_vertices - centroid
            
            # Convert axis-angle representation to rotation matrix
            R_w = Rotation.from_rotvec(obj_pose).as_matrix()
            
            # Build transformation matrix of mesh in world coordinates
            T_mesh = np.eye(4)
            T_mesh[:3, :3] = R_w  # No rotation applied, keeping it as identity matrix
            T_mesh[:3, 3] = obj_trans
            
            # Debug: Verify T_mesh
            # #print("T_mesh: ", T_mesh)

            # # Extract rotation and translation of camera from world coordinates
            # R_w_c = np.array(cam_params['rotation']).reshape(3, 3)
            # t_w_c = np.array(cam_params['translation']).reshape(3,)
            
            # # Build transformation matrix of camera in world coordinates
            # T_cam = np.eye(4)
            # T_cam[:3, :3] = R_w_c
            # T_cam[:3, 3] = t_w_c
            
            # # Debug: Verify T_cam
            # #print("T_cam: ", T_cam)

            # Ensure types are float64
            #T_cam = T_cam.astype(np.float64)
            T_mesh = T_mesh.astype(np.float64)

            # Calculate transformation matrix of mesh in camera frame
            # T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh
            T_mesh_in_cam = T_mesh

            # Debug: Verify T_mesh_in_cam
            ##print("T_mesh_in_cam: ", T_mesh_in_cam)
            
            # Transform the object's vertices using T_mesh_in_cam
            transformed_vertices = object_vertices
            transformed_vertices_homogeneous = T_mesh_in_cam @ np.vstack((transformed_vertices.T, np.ones(transformed_vertices.shape[0])))
            transformed_vertices = transformed_vertices_homogeneous[:3, :].T

            # Debug: Check transformed object
            # #print("Transformed vertices: ", transformed_vertices)

            # Update object mesh vertices
            object_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            
            # Extract new object translation in camera frame for further use if needed
            obj_trans_new_frame = T_mesh_in_cam[:3, 3]

            return object_mesh

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

        def interpolate_frames(all_data_frames, N):
            # Number of frames after = 1 + (N-1) * 2
            interpolated_frames = []

            for idx in range(len(all_data_frames)-1):
                frame1 = all_data_frames[idx]
                frame2 = all_data_frames[idx+1]

                # Original frame
                interpolated_frames.append(frame1)

                # Interpolated frames

                for i in range(1,N,1):
                    interpolated_frame = copy.deepcopy(frame1)
                    t = i / N  
                    interpolated_frame['pose'] = slerp_rotations(frame1['pose'], frame2['pose'], t)
                    interpolated_frame['trans'] = linear_interpolate(frame1['trans'], frame2['trans'], t)
                    interpolated_frame['obj_pose'] = slerp_rotations(frame1['obj_pose'], frame2['obj_pose'], t)
                    interpolated_frame['obj_trans'] = linear_interpolate(frame1['obj_trans'], frame2['obj_trans'], t)
                    
                    interpolated_frames.append(interpolated_frame)            

            # Adding the last original frame
            interpolated_frames.append(all_data_frames[-1])

            return interpolated_frames

        def project_frames(data_frames, timestamps, N):

            #print(len(data_frames))
            #print(len(timestamps))

            # Initialize a dictionary to hold lists for each camera
            cam_lists = {
                0: [],
                1: [],
                2: [],
                3: []
            }

            x_percent = 2.5  # Replace with the percentage you want, e.g., 50 for 50%

            # Calculate the number of frames to select
            total_frames = len(data_frames)
            frames_to_select = int(total_frames * (x_percent / 100))

            # Calculate start and end indices
            # start_index = (total_frames - frames_to_select) // 2
            # end_index = start_index + frames_to_select

            # Loop over the selected range of frames
            for idx in range(0, frames_to_select):
                input_frame = data_frames[idx]

                for cam_id in [0, 1, 2, 3]:
                    frame = copy.deepcopy(input_frame)
                    #print(f"\nProcessing frame {idx}: {frame}")
                    cam_params = load_config(cam_id, base_path_template, frame['date'])
                    intrinsics_cam, distortion_cam = load_intrinsics_and_distortion(cam_id, base_path_template)
                    transformed_smpl_pose, transformed_smpl_trans = transform_smpl_to_camera_frame(frame['pose'], frame['trans'], cam_params)
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

                    selected_file_path_smpl_trace = os.path.join(base_path_trace,label+f".{cam_id}.color.mp4.npz")

                    with np.load(selected_file_path_smpl_trace, allow_pickle=True) as data_smpl_trace:
                        outputs = data_smpl_trace['outputs'].item()  # Access the 'outputs' dictionary

                        pose_trace = outputs['smpl_thetas']
                        joints_trace = outputs['j3d'][:,:24,:] 
                        trans_trace = outputs['j3d'][:,0,:]
                        betas_trace = outputs['smpl_betas']
                        image_paths = data_smpl_trace['imgpaths']
                    
                        total_frames = int(pose_trace.shape[0])


                        def find_idx_global(timestamp, fps=30):
                            # Split the timestamp into seconds and milliseconds
                            parts = timestamp[1:].split('.')
                            seconds = int(parts[0])
                            milliseconds = int(parts[1])

                            # Convert the timestamp into a frame number
                            glb_idx = seconds * fps + int(round(milliseconds * fps / 1000))
                            #glb_idx = frame_number

                            return glb_idx

                        def find_idx_no_int(idx, N):
                            return math.floor(idx / N)

                        # The idx_
                        idx_no_int = find_idx_no_int(idx, N)
                        idx_global = find_idx_global(timestamps[idx_no_int])

                        if idx_global + 1 <= total_frames:
                            frame['img'] = image_paths[idx_global]
                            frame['pose_trace'] = pose_trace[idx_global,:]
                            frame['trans_trace'] = trans_trace[idx_global,:]
                            frame['betas_trace'] = betas_trace[idx_global,:]
                            frame['joints_trace'] = joints_trace[idx_global,:]

                            # Interpolation of pose_trace, trans_trace, joints_trace
                            # Interpolation possible from idx = 1 onward, for the previous value, every N = 2
                            # Indexes is even, 
                            # Update

                            if idx % N == 0 and idx >= 2:  # Check if idx is divisible by N
                                # Update the previous based on the second to last and last. Only linear interpolation as we deal with PC. At 1/2.
                                cam_lists[cam_id][-1]['joints_trace'] = linear_interpolate(cam_lists[cam_id][-N]['joints_trace'], frame['joints_trace'], 1/N)
                                cam_lists[cam_id][-1]['trans_trace'] = linear_interpolate(cam_lists[cam_id][-N]['trans_trace'], frame['trans_trace'], 1/N)
                                cam_lists[cam_id][-1]['pose_trace'] = slerp_rotations(cam_lists[cam_id][-N]['pose_trace'], frame['pose_trace'], 1/N)
                        else:
                            # Delete all frames
                            del frame
                
                    if 'frame' in locals():
                        cam_lists[cam_id].append(frame)

            scene_boundaries = []

            # Debug: #print sample from cam0_list after all operations
            #if cam_lists[0]:
                #print(f"\nSample from cam0_list prior to all operations: {cam_lists[0][0]}")

            # Gather components for normalization (here the normalization is over the whole test set)
            for cam_id in range(4):
                for idx in range(len(cam_lists[cam_id])):
                    # Flatten the joints and add to scene_boundaries
                    scene_boundaries.extend(np.array(cam_lists[cam_id][idx]['joints_trace']).flatten())
                    # Add obj_trans components to scene_boundaries
                    #scene_boundaries.extend(cam_lists[cam_id][idx]['obj_trans'].flatten())

            # Convert to numpy array for the min and max operations
            scene_boundaries_np = np.array(scene_boundaries)

            max_value = scene_boundaries_np.max()
            min_value = scene_boundaries_np.min()

            #print(f"\nMin value for normalization: {min_value}")
            #print(f"Max value for normalization: {max_value}")

            for cam_id in range(4):
                for idx in range(len(cam_lists[cam_id])):
                    cam_lists[cam_id][idx]['norm_obj_trans'] = normalize(cam_lists[cam_id][idx]['obj_trans'], 0, 2*np.pi, min_value, max_value)
                    cam_lists[cam_id][idx]['norm_joints'] = normalize(cam_lists[cam_id][idx]['joints'], 0, 2*np.pi, min_value, max_value)
                    cam_lists[cam_id][idx]['norm_joints_trace'] = normalize(cam_lists[cam_id][idx]['joints_trace'], 0, 2*np.pi, min_value, max_value)

            #if cam_lists[0]:
                #print(f"\nSample from cam0_list after normalizing: {cam_lists[0][0]}")
                
            # Unroll angle hierarchy
            #print("\nUnrolling angle hierarchy...") 
            for cam_id in range(4):
                for idx in range(len(cam_lists[cam_id])):
                    cam_lists[cam_id][idx]['unrolled_pose'] = process_pose_params(cam_lists[cam_id][idx]['pose'])
                    cam_lists[cam_id][idx]['unrolled_pose_trace'] = process_pose_params(cam_lists[cam_id][idx]['pose_trace'])

            #if cam_lists[0]:
                #print(f"\nSample from cam0_list after unrolling: {cam_lists[0][0]}")

            # Positional encoding
            #print("\nApplying positional encoding...")
            L = wandb.config.L
            for cam_id in range(4):
                for idx in range(len(cam_lists[cam_id])):
                    cam_lists[cam_id][idx]['enc_norm_joints'] = gamma(cam_lists[cam_id][idx]['norm_joints'], L)
                    cam_lists[cam_id][idx]['enc_unrolled_pose'] = gamma(cam_lists[cam_id][idx]['unrolled_pose'], L)
                    cam_lists[cam_id][idx]['enc_obj_pose'] = gamma(cam_lists[cam_id][idx]['obj_pose'], L)
                    cam_lists[cam_id][idx]['enc_norm_obj_trans'] = gamma(cam_lists[cam_id][idx]['norm_obj_trans'], L)
                    cam_lists[cam_id][idx]['enc_norm_joints_trace'] = gamma(cam_lists[cam_id][idx]['norm_joints_trace'], L)
                    cam_lists[cam_id][idx]['enc_unrolled_pose_trace'] = gamma(cam_lists[cam_id][idx]['unrolled_pose_trace'], L)

            # Debug: #print sample from cam0_list after all operations
            #if cam_lists[0]:
                #print(f"\nSample from cam0_list after all operations: {cam_lists[0][0]}")

            return [cam_lists[0], cam_lists[1], cam_lists[2], cam_lists[3]]

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

        def evaluate_camera(y_hat_stage3_pos, y_hat_stage3_trans, y_stage3_pos, y_stage3_trans, obj_template_path):
            """
            Evaluate the performance metrics for a given camera.

            Parameters:
            cam_id (int): Camera ID.
            y_hat_stage3_pos (numpy array): Predicted position.
            y_hat_stage3_trans (numpy array): Predicted transformation.
            y_stage3_pos (numpy array): Ground truth position.
            y_stage3_trans (numpy array): Ground truth transformation.
            obj_template_path (str): Path to the object template.

            Returns:
            dict: Dictionary containing ADD, ADD-S, and CD values.
            """

            def compute_cd(candidate_vertices, GT_vertices):
                # Convert lists to numpy arrays for efficient computation
                A = np.array(GT_vertices)
                B = np.array(candidate_vertices)

                # Compute squared distances from each point in A to the closest point in B, and vice versa
                A_B_dist = np.sum([min(np.sum((a - B)**2, axis=1)) for a in A])
                B_A_dist = np.sum([min(np.sum((b - A)**2, axis=1)) for b in B])

                # Compute the Chamfer Distance
                chamfer_distance = A_B_dist / len(A) + B_A_dist / len(B)

                return chamfer_distance

            def add_err(pred_pts, gt_pts):
                """
                Average Distance of Model Points for objects with no indistinguishable views
                - by Hinterstoisser et al. (ACCV 2012).
                """
                #   pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
                #   gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
                e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
                return e

            def adi_err(pred_pts, gt_pts):
                """
                @pred: 4x4 mat
                @gt:
                @model: (N,3)
                """
                # = (pred@to_homo(model_pts).T).T[:,:3]
                #gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
                nn_index = cKDTree(pred_pts)
                nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
                e = nn_dists.mean()
                return e
            
            # only the first from the batch
            transformed_object = plot_obj_in_camera_frame(y_hat_stage3_pos[0].cpu().numpy(), y_hat_stage3_trans[0].cpu().numpy(), obj_template_path)
            GT_obj = plot_obj_in_camera_frame(y_stage3_pos[0].cpu().numpy(), y_stage3_trans[0].cpu().numpy(), obj_template_path)

            # Convert the meshes to point clouds
            GT_obj_pcd = o3d.geometry.PointCloud()
            GT_obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(GT_obj.vertices))

            candidate_obj_pcd = o3d.geometry.PointCloud()
            candidate_obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(transformed_object.vertices))

            # Convert to numpy arrays
            GT_obj_np = np.asarray(GT_obj_pcd.points)
            candidate_obj_np = np.asarray(candidate_obj_pcd.points)

            num_points = GT_obj_np.shape[0]
            num_sampled_points = 10  # Example: 10 sampled points

            np.random.seed(0)
            random_indices = np.random.choice(num_points, num_sampled_points, replace=False)

            GT_vertices = GT_obj_np[random_indices]
            candidate_vertices = candidate_obj_np[random_indices]

            add = add_err(candidate_vertices, GT_vertices)
            add_s = adi_err(candidate_vertices, GT_vertices)
            cd = compute_cd(candidate_vertices, GT_vertices)

            return add, add_s, cd

        def compute_auc(rec, max_val=0.1):
            if len(rec) == 0:
                return 0
            rec = np.sort(np.array(rec))
            n = len(rec)
            print(n)
            prec = np.arange(1, n + 1) / float(n)
            rec = rec.reshape(-1)
            prec = prec.reshape(-1)
            index = np.where(rec < max_val)[0]
            rec = rec[index]
            prec = prec[index]

            if len(prec) == 0:
                return 0

            mrec = [0, *list(rec), max_val]
            # Only add prec[-1] if prec is not empty
            mpre = [0, *list(prec)] + ([prec[-1]] if len(prec) > 0 else [])

            for i in range(1, len(mpre)):
                mpre[i] = max(mpre[i], mpre[i - 1])
            mpre = np.array(mpre)
            mrec = np.array(mrec)
            i = np.where(mrec[1:] != mrec[:len(mrec) - 1])[0] + 1
            ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val
            return ap

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
                return len(self.cam_data[3])

            def __getitem__(self, idx):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Extracting the required keys and converting them to tensors
                def extract_and_convert(cam_id):
                    # Keys for encoding of previous step 
                    # keys = ['enc_unrolled_pose', 'enc_norm_joints', 'prev_enc_obj_pose', 'prev_enc_norm_obj_trans', 'obj_pose', 'obj_trans']
                    keys = [ SMPL_pose, SMPL_joints, 'prev_obj_pose', 'prev_obj_trans', OBJ_pose, OBJ_trans]
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

                for idx, data in enumerate(self.dataset.cam_data[3]):
                    if data['scene'] in self.split['train']:
                        self.train_indices.append(idx)
                        train_identifiers.append(data['scene'])
                    elif data['scene'] in self.split['test']:
                        self.test_indices.append(idx)
                        test_identifiers.append(data['scene'])

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
                self.all_ADD_values = {0: [], 1: [], 2: [], 3: []}
                self.all_ADD_S_values = {0: [], 1: [], 2: [], 3: []}
                self.all_CD_values = {0: [], 1: [], 2: [], 3: []}
                self.max_th = 0.10
            
            def forward(self, cam_data):

                smpl_pose, smpl_joints, obj_pose, obj_trans, _, _ = cam_data

                x_stage1_pose = self.model1(smpl_pose)
                #x_stage1_joints = self.model1(smpl_joints.reshape(-1,72)) Needed for 24,3 shape of joints
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
                    pos_loss = lambda_2 * axis_angle_loss(y_hat_pos, y_pos) + \
                            lambda_1 * F.mse_loss(y_hat_pos, y_pos) + \
                            lambda_2 * (1 - F.cosine_similarity(y_hat_pos, y_pos)) + \
                            lambda_3 * smooth_sign_loss(y_hat_pos, y_pos) + \
                            lambda_4 * geodesic_loss(y_hat_pos, y_pos)

                    
                    trans_loss = lambda_1 * F.mse_loss(y_hat_trans, y_trans) + \
                                lambda_2 * (1 - F.cosine_similarity(y_hat_trans, y_trans)) + \
                                lambda_3 * smooth_sign_loss(y_hat_trans, y_trans)
                    
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
                obj_template_path = "/scratch_net/biwidl307_second/lgermano/behave/objects/boxmedium/boxmedium.obj"

                # Existing code for camera 0
                add, add_s, cd = evaluate_camera(y_hat_stage3_pos_cam0, y_hat_stage3_trans_cam0, y_stage3_pos_cam0, y_stage3_trans_cam0, obj_template_path)
                self.all_ADD_values[0].append(add)
                self.all_ADD_S_values[0].append(add_s)
                self.all_CD_values[0].append(cd)

                # Cam 1
                add, add_s, cd = evaluate_camera(y_hat_stage3_pos_cam1, y_hat_stage3_trans_cam1, y_stage3_pos_cam1, y_stage3_trans_cam1, obj_template_path)
                self.all_ADD_values[1].append(add)
                self.all_ADD_S_values[1].append(add_s)
                self.all_CD_values[1].append(cd)

                # Code for camera 2
                add, add_s, cd = evaluate_camera(y_hat_stage3_pos_cam2, y_hat_stage3_trans_cam2, y_stage3_pos_cam2, y_stage3_trans_cam2, obj_template_path)
                self.all_ADD_values[2].append(add)
                self.all_ADD_S_values[2].append(add_s)
                self.all_CD_values[2].append(cd)

                # Code for camera 3
                add, add_s, cd = evaluate_camera(y_hat_stage3_pos_cam3, y_hat_stage3_trans_cam3, y_stage3_pos_cam3, y_stage3_trans_cam3, obj_template_path)
                self.all_ADD_values[3].append(add)
                self.all_ADD_S_values[3].append(add_s)
                self.all_CD_values[3].append(cd)

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
                    # model_save_path = f'/scratch_net/biwidl307_second/lgermano/H2O/trained_models/model_{wandb.run.name}_epoch_{self.current_epoch}.pt'
                    model_save_path = f'/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/model_{wandb.run.name}_epoch_{self.current_epoch}.pt'
                    torch.save(self.state_dict(), model_save_path)
                    print(f'Model saved to {model_save_path}')

                self.validation_losses = []  # reset for the next epoch
                self.lr_scheduler.step(avg_val_loss)  # Update

                for cam_id in range(4):
                    auc_ADD = compute_auc(np.array(self.all_ADD_values[cam_id]), self.max_th) * 100
                    auc_ADD_S = compute_auc(np.array(self.all_ADD_S_values[cam_id]), self.max_th) * 100
                    cd_mean = sum(self.all_CD_values[cam_id]) / len(self.all_CD_values[cam_id])

                    print(f"AUC boxmedium for camera {cam_id} - ADD: {auc_ADD:.2f}%, ADD-S: {auc_ADD_S:.2f}%, CD[m]: {cd_mean:.5f}")
                    # wandb.log({f"AUC boxmedium for camera {cam_id}, ADD": f"{auc_ADD:.2f}%"})
                    # wandb.log({f"AUC boxmedium for camera {cam_id}, ADD-S": f"{auc_ADD_S:.2f}%"})
                    # wandb.log({f"Boxmedium for camera {cam_id}, CD[m]": f"{cd_mean:.2f}"})
                    wandb.log({f"AUC boxmedium for camera {cam_id}, ADD": auc_ADD})
                    wandb.log({f"AUC boxmedium for camera {cam_id}, ADD-S": auc_ADD_S})
                    wandb.log({f"Boxmedium for camera {cam_id}, CD[m]": cd_mean})


                self.all_ADD_values = {0: [], 1: [], 2: [], 3: []}
                self.all_ADD_S_values = {0: [], 1: [], 2: [], 3: []}
                self.all_CD_values = {0: [], 1: [], 2: [], 3: []}

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
        base_path_annotations = "/scratch_net/biwidl307_second/lgermano/behave/behave-30fps-params-v1"
        #base_path_trace = "/scratch_net/biwidl307_second/lgermano/TRACETRACE_results"
        base_path_trace = "/srv/beegfs02/scratch/3dhumanobjint/data/TRACE_results"
        base_path_template = "/scratch_net/biwidl307_second/lgermano/behave"


        # Check if the data has already been saved
        if os.path.exists(data_file_path) and False:
            # Load the saved data
            with open(data_file_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            # # Create a dataset

            # # Test sequences should not be interpolated ('Date03'), set N=1, else N=2
            N = 2

            wandb.run.name = name

            base_path = "/scratch_net/biwidl307_second/lgermano/behave"
            #labels = sorted([label.split('.')[0] for label in os.listdir(base_path_trace) if 'boxlarge' in label and '.color.mp4.npz' in label and 'Date03' not in label and 'boxmedium' not in label])
            # labels = sorted(set([label.split('.')[0] for label in os.listdir(base_path_trace) if '.color.mp4.npz' in label and 'boxmedium' not in label and 'Date03' not in label]))
            # print(labels)

            # dataset = []

            # for label in labels:
            #     print("Processing label:", label)
            #     selected_file_path_obj = os.path.join(base_path_annotations, label, "object_fit_all.npz")
            #     selected_file_path_smpl = os.path.join(base_path_annotations, label, "smpl_fit_all.npz")

            #     print("Object file path:", selected_file_path_obj)
            #     print("SMPL file path:", selected_file_path_smpl)

            #     all_data_frames = []

            #     with np.load(selected_file_path_obj, allow_pickle=True) as data_obj:
            #         print("Loading object data")
            #         obj_pose = data_obj['angles']
            #         obj_trans = data_obj['trans']
            #         timestamps = data_obj['frame_times']
            #         print("Object data loaded. Shape:", obj_pose.shape)

            #     with np.load(selected_file_path_smpl, allow_pickle=True) as data_smpl:
            #         print("Loading SMPL data")
            #         pose = data_smpl['poses'][:,:72]  # SMPL model
            #         trans = data_smpl['trans']
            #         betas = data_smpl['betas']
            #         print("SMPL data loaded. Shape:", pose.shape)

            #     for idx in range(trans.shape[0]):
            #         print(f"Processing frame {idx}")
            #         frame_data = {}
            #         obj_name = label.split('_')[2]
            #         frame_data['obj_template_path'] = os.path.join(base_path_template, "objects", obj_name, obj_name + ".obj")
            #         frame_data['scene'] = label
            #         frame_data['date'] = label.split('_')[0]
            #         frame_data['pose'] = pose[idx,:]
            #         frame_data['trans'] = trans[idx,:]
            #         frame_data['betas'] = betas[idx,:]
            #         frame_data['obj_pose'] = obj_pose[idx,:]
            #         frame_data['obj_trans'] = obj_trans[idx,:]

            #         all_data_frames.append(frame_data)

            #     # Assuming interpolate_frames and project_frames are defined elsewhere in your script
            #     all_data_frames_int = interpolate_frames(all_data_frames, N)
            #     print("Interpolation done. Length of interpolated frames:", len(all_data_frames_int))
            #     del all_data_frames

            #     dataset_label = project_frames(all_data_frames_int, timestamps, N)
            #     print("Projection done. Length of projected frames:", len(dataset_label))
            #     del all_data_frames_int

            #     data_file_path = f'/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_int_1frame/{label}.pkl'
            #     print(f"Saving data to {data_file_path}")
            #     with open(data_file_path, 'wb') as f:
            #         pickle.dump(dataset_label, f)
            #     print(f"Saved data for {label} to {data_file_path}")

            #     del dataset_label
            #     gc.collect()

        # Include now Date03. No processing. 

        # labels=['Date05_Sub06_boxmedium', 'Date04_Sub05_boxmedium', 'Date07_Sub08_boxmedium', 'Date03_Sub03_boxmedium', \
        # 'Date03_Sub04_boxmedium', 'Date03_Sub05_boxmedium', 'Date02_Sub02_boxmedium_hand', 'Date07_Sub04_boxmedium', \
        # 'Date06_Sub07_boxmedium', 'Date01_Sub01_boxmedium_hand']

        # labels=['Date05_Sub06_boxmedium', 'Date07_Sub08_boxmedium', 'Date03_Sub03_boxmedium', \
        # 'Date03_Sub04_boxmedium', 'Date03_Sub05_boxmedium', 'Date02_Sub02_boxmedium_hand', 'Date07_Sub04_boxmedium', \
        # 'Date06_Sub07_boxmedium', 'Date01_Sub01_boxmedium_hand', 'Date03_Sub03_boxlarge', 'Date03_Sub03_boxlarge', \
        # 'Date03_Sub03_boxlarge', 'Date03_Sub03_boxlarge']

        # labels = list(sorted(set([label.split('.')[0] for label in os.listdir(base_path_trace) if '.color.mp4.npz' in label and 'Date03' not in label])))
        # labels = labels + ['Date03_Sub03_boxmedium', 'Date03_Sub04_boxmedium', 'Date03_Sub05_boxmedium']

        # labels = [
        #     "Date01_Sub01_boxlarge_hand",
        #     "Date01_Sub01_boxmedium_hand",
        #     "Date02_Sub02_boxmedium_hand",
        #     #"Date03_Sub03_boxlarge",
        #     "Date03_Sub03_boxmedium",
        #     "Date03_Sub04_boxmedium",
        #     "Date03_Sub05_boxmedium",
        #     "Date04_Sub05_boxlarge",
        #     "Date04_Sub05_boxmedium",
        #     "Date05_Sub06_boxmedium",
        #     "Date06_Sub07_boxmedium",
        #     "Date07_Sub04_boxmedium",
        #     "Date07_Sub08_boxmedium"
        # ]

        labels = [
            "Date01_Sub01_boxmedium_hand", "Date04_Sub05_boxsmall", "Date05_Sub06_toolbox", "Date07_Sub04_boxlong",
            "Date02_Sub02_boxmedium_hand", "Date04_Sub05_boxtiny", "Date06_Sub07_boxlarge", "Date07_Sub04_boxmedium",
            "Date03_Sub03_boxmedium", "Date04_Sub05_toolbox", "Date06_Sub07_boxlong", "Date07_Sub04_boxsmall",
            "Date03_Sub04_boxmedium", "Date05_Sub06_boxlarge", "Date06_Sub07_boxmedium", "Date07_Sub04_boxtiny",
            "Date03_Sub05_boxmedium", "Date05_Sub06_boxlong", "Date06_Sub07_boxsmall", "Date07_Sub08_boxmedium",
            "Date04_Sub05_boxlarge", "Date05_Sub06_boxmedium", "Date06_Sub07_boxtiny",
            "Date04_Sub05_boxlong", "Date05_Sub06_boxsmall", "Date06_Sub07_toolbox",
            "Date05_Sub06_boxtiny", "Date07_Sub04_boxlarge"
        ]

        print("\nTraining on:", labels)

        # Splitting the labels for validation and training
        val_labels = [label for label in labels if label.startswith("Date03")]
        train_labels = [label for label in labels if label not in val_labels]
        train_set = val_labels + train_labels

        dataset = []

        #Processing camera data more efficiently
        cam_data = {0: [], 1: [], 2: [], 3: []}

        for label in labels:
            print("Appending label",label)
            #data_file_path = f'/scratch_net/biwidl307_second/lgermano/H2O/datasets/30fps_int_1frame/{label}.pkl'
            data_file_path = f'/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_int_1frame/{label}.pkl'

            with open(data_file_path, 'rb') as f:
                dataset = pickle.load(f)

            for cam_id in range(4):
                for idx in range(len(dataset[cam_id])):
                    # TODO: add obj_template_path when diff object are present
                    #data_dict = {key: dataset[cam_id][idx][key] for key in dataset[cam_id][idx] if key in ['enc_unrolled_pose_trace', 'enc_norm_joints_trace', 'obj_pose', 'obj_trans', 'scene']}
                    data_dict = {key: dataset[cam_id][idx][key] for key in dataset[cam_id][idx] if key in [SMPL_pose, SMPL_joints, OBJ_pose, OBJ_trans, 'scene']}
                    
                    # Use previous frame data for 'prev_' fields
                    if idx != 0:
                        data_dict.update({
                            #'prev_enc_obj_pose': dataset[cam_id][idx-1]['enc_obj_pose'],
                            #'prev_enc_norm_obj_trans': dataset[cam_id][idx-1]['enc_norm_obj_trans'],
                            'prev_obj_pose': dataset[cam_id][idx-1][OBJ_pose],
                            'prev_obj_trans':  dataset[cam_id][idx-1][OBJ_trans],
                        })

                        cam_data[cam_id].append(data_dict)
                    else:
                        data_dict.update({
                                #'prev_enc_obj_pose': dataset[cam_id][idx-1]['enc_obj_pose'],
                                #'prev_enc_norm_obj_trans': dataset[cam_id][idx-1]['enc_norm_obj_trans'],
                                'prev_obj_pose': dataset[cam_id][idx][OBJ_pose],
                                'prev_obj_trans':  dataset[cam_id][idx][OBJ_trans],
                            })

                        cam_data[cam_id].append(data_dict)
            
            del dataset

        dataset_grouped = BehaveDataset(cam_data)

        path_to_file = "/scratch_net/biwidl307_second/lgermano/behave/split.json"
        split_dict = load_split_from_path(path_to_file)

        data_module = BehaveDataModule(dataset_grouped, split=split_dict, batch_size=BATCH_SIZE)

        # Save the data module with reduced memory footprint
        save_file_name = f"{wandb.run.name}.pt"
        #data_file_path = '/scratch_net/biwidl307_second/lgermano/H2O/data_module'
        data_file_path = '/srv/beegfs02/scratch/3dhumanobjint/data/H2O/data_module'
        full_save_path = os.path.join(data_file_path, save_file_name)
        torch.save(data_module, full_save_path)

        #####################################
        # Train

        # Load any data module
        # save_file_name = "boxmedium_pose_joints_obj_pose_obj_trans.pt"

        # # Define the local path where the data module will be saved
        # data_file_path = '/scratch_net/biwidl307_second/lgermano/H2O/data_module'
        # full_save_path = os.path.join(data_file_path, save_file_name)
        
        # # Load the data module back to a variable named data_module
        # data_module = torch.load(full_save_path)

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
        # Sizes depend on settings
        input_dim = 72 * wandb.config.L * 2 if "enc" in SMPL_pose or "enc" in SMPL_joints else 72
        input_stage3 = 3 * wandb.config.L * 2 if "enc" in OBJ_pose or "enc" in OBJ_trans else 6

        
        # Invariant
        output_stage1 = 256
        input_stage2 = 512
        output_stage2 = 3
        output_dim = 3
       
        print(f"Wandb run name:{wandb.run.name}")
        model_combined = CombinedMLP(input_dim, output_stage1, input_stage2, output_stage2, input_stage3, output_dim)

        # Move the model to device
        model_combined.to(device)

        # #Specify the path to the checkpoint
        # model_path = f"/scratch_net/biwidl307_second/lgermano/H2O/trained_models/model_leafy-smoke-2619_epoch_16.pt"
        # #model_path = f"/srv/beegfs02/scratch/3dhumanobjint/data"

        # #Load the state dict from the checkpoint into the model
        # checkpoint = torch.load(model_path, map_location=device)
        # model_combined.load_state_dict(checkpoint)

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
            # filename = f"/scratch_net/biwidl307_second/lgermano/H2O/trained_models/{wandb.run.name}.pt"
            filename = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/{wandb.run.name}.pt"

            # Save the model
            torch.save(model_combined, filename)

        # Finish the current W&B run
        wandb.finish()

    #After all trials, #print the best set of hyperparameters
    print("Best Validation Loss:", best_overall_avg_loss_val)
    print("Best Hyperparameters:", best_params)


