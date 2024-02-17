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

# from memory_profiler import profile
import pdb
from pytorch_lightning.loggers import WandbLogger
# from behave_dataset import BehaveDatasetOffset, BehaveDatasetOffset2, BehaveDataModule


def load_intrinsics_and_distortion(camera_id, base_path):
    calib_path = os.path.join(base_path, "calibs", "intrinsics", str(camera_id), "calibration.json")
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
        color_intrinsics = calib_data["color"]
    return {
        "fx": color_intrinsics["fx"],
        "fy": color_intrinsics["fy"],
        "cx": color_intrinsics["cx"],
        "cy": color_intrinsics["cy"],
    }, {
        "k1": color_intrinsics["k1"],
        "k2": color_intrinsics["k2"],
        "k3": color_intrinsics["k3"],
        "p1": color_intrinsics["p1"],
        "p2": color_intrinsics["p2"],
    }


def plot_obj_in_camera_frame(obj_pose, obj_trans, obj_template_path):
    # Load obj template
    # object_template = "/scratch_net/biwidl307_second/lgermano/behave/objects/stool/stool.obj"
    object_mesh = o3d.io.read_triangle_mesh(obj_template_path)
    object_vertices = np.asarray(object_mesh.vertices)

    # Debug: ##print object vertices before any transformation
    ###print("Object vertices before any transformation: ", object_vertices)

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
    # ##print("T_mesh: ", T_mesh)

    # # Extract rotation and translation of camera from world coordinates
    # R_w_c = np.array(cam_params['rotation']).reshape(3, 3)
    # t_w_c = np.array(cam_params['translation']).reshape(3,)

    # # Build transformation matrix of camera in world coordinates
    # T_cam = np.eye(4)
    # T_cam[:3, :3] = R_w_c
    # T_cam[:3, 3] = t_w_c

    # # Debug: Verify T_cam
    # ##print("T_cam: ", T_cam)

    # Ensure types are float64
    # T_cam = T_cam.astype(np.float64)
    T_mesh = T_mesh.astype(np.float64)

    # Calculate transformation matrix of mesh in camera frame
    # T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh
    T_mesh_in_cam = T_mesh

    # Debug: Verify T_mesh_in_cam
    ###print("T_mesh_in_cam: ", T_mesh_in_cam)

    # Transform the object's vertices using T_mesh_in_cam
    transformed_vertices = object_vertices
    transformed_vertices_homogeneous = T_mesh_in_cam @ np.vstack(
        (transformed_vertices.T, np.ones(transformed_vertices.shape[0]))
    )
    transformed_vertices = transformed_vertices_homogeneous[:3, :].T

    # Debug: Check transformed object
    # ##print("Transformed vertices: ", transformed_vertices)

    # Update object mesh vertices
    object_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

    # Extract new object translation in camera frame for further use if needed
    obj_trans_new_frame = T_mesh_in_cam[:3, 3]

    return object_mesh


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_config(camera_id, base_path, Date="Date07"):
    config_path = os.path.join(base_path, "calibs", Date, "config", str(camera_id), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_split_from_path(path):
    with open(path, "r") as file:
        split_dict = json.load(file)
    return split_dict


def linear_interpolate(value1, value2, i):
    return value1 + (i / 3) * (value2 - value1)


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
    res_quat = (np.sin((1.0 - t) * omega) / so) * q0 + (np.sin(t * omega) / so) * q1

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

    for idx in range(len(all_data_frames) - 1):
        frame1 = all_data_frames[idx]
        frame2 = all_data_frames[idx + 1]

        # Original frame
        interpolated_frames.append(frame1)

        # Interpolated frames

        for i in range(1, N, 1):
            interpolated_frame = copy.deepcopy(frame1)
            t = i / N
            interpolated_frame["pose"] = slerp_rotations(frame1["pose"], frame2["pose"], t)
            interpolated_frame["trans"] = linear_interpolate(frame1["trans"], frame2["trans"], t)
            interpolated_frame["obj_pose"] = slerp_rotations(frame1["obj_pose"], frame2["obj_pose"], t)
            interpolated_frame["obj_trans"] = linear_interpolate(frame1["obj_trans"], frame2["obj_trans"], t)

            interpolated_frames.append(interpolated_frame)

    # Adding the last original frame
    interpolated_frames.append(all_data_frames[-1])

    return interpolated_frames


def project_frames(data_frames, timestamps, N, base_path_template, base_path_trace, label):
    print(len(data_frames))
    print(len(timestamps))

    # Initialize a dictionary to hold lists for each camera
    cam_lists = {0: [], 1: [], 2: [], 3: []}

    x_percent = 100  # Replace with the percentage you want, e.g., 50 for 50%

    # Calculate the number of frames to select
    total_frames = len(data_frames)
    frames_to_select = int(total_frames * (x_percent / 100))

    # # Calculate start and end indices
    # start_index = (total_frames - frames_to_select) // 2
    # end_index = start_index + frames_to_select

    # Loop over the selected range of frames
    for idx in range(0, frames_to_select):
        input_frame = data_frames[idx]

        for cam_id in [0, 1, 2, 3]:
            frame = copy.deepcopy(input_frame)
            print(f"\nProcessing frame {idx}: {frame}")
            cam_params = load_config(cam_id, base_path_template, frame["date"])
            intrinsics_cam, distortion_cam = load_intrinsics_and_distortion(cam_id, base_path_template)
            transformed_smpl_pose, transformed_smpl_trans = transform_smpl_to_camera_frame(
                frame["pose"], frame["trans"], cam_params
            )
            frame["SMPL_pose"] = transformed_smpl_pose
            frame["trans"] = transformed_smpl_trans
            joints = render_smpl(transformed_smpl_pose, transformed_smpl_trans, frame["betas"])
            joints_numpy = [joint.cpu().numpy() for joint in joints]
            frame["SMPL_joints"] = joints_numpy
            transformed_obj_pose, transformed_obj_trans = transform_object_to_camera_frame(
                frame["obj_pose"], frame["obj_trans"], cam_params
            )
            frame["OBJ_pose"] = transformed_obj_pose
            frame["OBJ_trans"] = transformed_obj_trans
            distances = np.asarray([np.linalg.norm(transformed_obj_trans - joint) for joint in joints_numpy])
            frame["distances"] = distances

            # selected_file_path_smpl_trace = os.path.join(base_path_trace,label+f".{cam_id}.color.mp4.npz")

            # with np.load(selected_file_path_smpl_trace, allow_pickle=True) as data_smpl_trace:
            #     outputs = data_smpl_trace['outputs'].item()  # Access the 'outputs' dictionary

            #     pose_trace = outputs['smpl_thetas']
            #     joints_trace = outputs['j3d'][:,:24,:]
            #     trans_trace = outputs['j3d'][:,0,:]
            #     betas_trace = outputs['smpl_betas']
            #     image_paths = data_smpl_trace['imgpaths']

            #     total_frames = int(pose_trace.shape[0])

            #     def find_idx_global(timestamp, fps=30):
            #         # Split the timestamp into seconds and milliseconds
            #         parts = timestamp[1:].split('.')
            #         seconds = int(parts[0])
            #         milliseconds = int(parts[1])

            #         # Convert the timestamp into a frame number
            #         glb_idx = seconds * fps + int(round(milliseconds * fps / 1000))
            #         #glb_idx = frame_number

            #         return glb_idx

            #     def find_idx_no_int(idx, N):
            #         return math.floor(idx / N)

            #     # The idx_
            #     idx_no_int = find_idx_no_int(idx, N)
            #     idx_global = find_idx_global(timestamps[idx_no_int])

            #     if idx_global + 1 <= total_frames:
            #         frame['img'] = image_paths[idx_global]
            #         frame['pose_trace'] = pose_trace[idx_global,:]
            #         frame['trans_trace'] = trans_trace[idx_global,:]
            #         frame['betas_trace'] = betas_trace[idx_global,:]
            #         frame['joints_trace'] = joints_trace[idx_global,:]

            #         # Interpolation of pose_trace, trans_trace, joints_trace
            #         # Interpolation possible from idx = 1 onward, for the previous value, every N = 2
            #         # Indexes is even,
            #         # Update

            #         if idx % N == 0 and idx >= 2:  # Check if idx is divisible by N
            #             # Update the previous based on the second to last and last. Only linear interpolation as we deal with PC. At 1/2.
            #             cam_lists[cam_id][-1]['joints_trace'] = linear_interpolate(cam_lists[cam_id][-N]['joints_trace'], frame['joints_trace'], 1/N)
            #             cam_lists[cam_id][-1]['trans_trace'] = linear_interpolate(cam_lists[cam_id][-N]['trans_trace'], frame['trans_trace'], 1/N)
            #             cam_lists[cam_id][-1]['pose_trace'] = slerp_rotations(cam_lists[cam_id][-N]['pose_trace'], frame['pose_trace'], 1/N)
            #     else:
            #         # Delete all frames
            #         del frame

            # if "frame" in locals():
            cam_lists[cam_id].append(frame)

    # scene_boundaries = []

    # # Debug: ##print sample from cam0_list after all operations
    # #if cam_lists[0]:
    #     ##print(f"\nSample from cam0_list prior to all operations: {cam_lists[0][0]}")

    # # Gather components for normalization (here the normalization is over the whole test set)
    # for cam_id in range(4):
    #     for idx in range(len(cam_lists[cam_id])):
    #         # Flatten the joints and add to scene_boundaries
    #         scene_boundaries.extend(np.array(cam_lists[cam_id][idx]['joints_trace']).flatten())
    #         # Add obj_trans components to scene_boundaries
    #         #scene_boundaries.extend(cam_lists[cam_id][idx]['obj_trans'].flatten())

    # # Convert to numpy array for the min and max operations
    # scene_boundaries_np = np.array(scene_boundaries)

    # max_value = scene_boundaries_np.max()
    # min_value = scene_boundaries_np.min()

    # ##print(f"\nMin value for normalization: {min_value}")
    # ##print(f"Max value for normalization: {max_value}")

    # for cam_id in range(4):
    #     for idx in range(len(cam_lists[cam_id])):
    #         cam_lists[cam_id][idx]['norm_obj_trans'] = normalize(cam_lists[cam_id][idx]['obj_trans'], 0, 2*np.pi, min_value, max_value)
    #         cam_lists[cam_id][idx]['norm_joints'] = normalize(cam_lists[cam_id][idx]['joints'], 0, 2*np.pi, min_value, max_value)
    #         cam_lists[cam_id][idx]['norm_joints_trace'] = normalize(cam_lists[cam_id][idx]['joints_trace'], 0, 2*np.pi, min_value, max_value)

    # #if cam_lists[0]:
    #     ##print(f"\nSample from cam0_list after normalizing: {cam_lists[0][0]}")

    # # Unroll angle hierarchy
    # ##print("\nUnrolling angle hierarchy...")
    # for cam_id in range(4):
    #     for idx in range(len(cam_lists[cam_id])):
    #         cam_lists[cam_id][idx]['unrolled_pose'] = process_pose_params(cam_lists[cam_id][idx]['pose'])
    #         cam_lists[cam_id][idx]['unrolled_pose_trace'] = process_pose_params(cam_lists[cam_id][idx]['pose_trace'])

    # #if cam_lists[0]:
    #     ##print(f"\nSample from cam0_list after unrolling: {cam_lists[0][0]}")

    # # Positional encoding
    # ##print("\nApplying positional encoding...")
    # L = wandb.config.L
    # for cam_id in range(4):
    #     for idx in range(len(cam_lists[cam_id])):
    #         cam_lists[cam_id][idx]['enc_norm_joints'] = gamma(cam_lists[cam_id][idx]['norm_joints'], L)
    #         cam_lists[cam_id][idx]['enc_unrolled_pose'] = gamma(cam_lists[cam_id][idx]['unrolled_pose'], L)
    #         cam_lists[cam_id][idx]['enc_obj_pose'] = gamma(cam_lists[cam_id][idx]['obj_pose'], L)
    #         cam_lists[cam_id][idx]['enc_norm_obj_trans'] = gamma(cam_lists[cam_id][idx]['norm_obj_trans'], L)
    #         cam_lists[cam_id][idx]['enc_norm_joints_trace'] = gamma(cam_lists[cam_id][idx]['norm_joints_trace'], L)
    #         cam_lists[cam_id][idx]['enc_unrolled_pose_trace'] = gamma(cam_lists[cam_id][idx]['unrolled_pose_trace'], L)

    # # Debug: ##print sample from cam0_list after all operations
    # #if cam_lists[0]:
    #     ##print(f"\nSample from cam0_list after all operations: {cam_lists[0][0]}")

    return [cam_lists[0], cam_lists[1], cam_lists[2], cam_lists[3]]


def transform_smpl_to_camera_frame(pose, trans, cam_params):
    # Convert axis-angle representation to rotation matrix
    R_w = Rotation.from_rotvec(pose[:3]).as_matrix()

    # Build transformation matrix of mesh in world coordinates
    T_mesh = np.eye(4)
    T_mesh[:3, :3] = R_w
    T_mesh[:3, 3] = trans

    # Extract rotation and translation of camera from world coordinates
    R_w_c = np.array(cam_params["rotation"]).reshape(3, 3)
    t_w_c = np.array(cam_params["translation"]).reshape(
        3,
    )

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
    R_w_c = np.array(cam_params["rotation"]).reshape(3, 3)
    t_w_c = np.array(cam_params["translation"]).reshape(
        3,
    )

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
    ##print("Start of render_smpl function.")

    batch_size = 1
    ##print(f"batch_size: {batch_size}")

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="male",
        model_root="/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch/native/models/",
    )
    ##print("SMPL_Layer created.")

    # Process pose parameters
    pose_params_start = torch.tensor(transformed_pose[:3], dtype=torch.float32)
    pose_params_rest = torch.tensor(transformed_pose[3:72], dtype=torch.float32)
    pose_params_rest[-6:] = 0
    pose_params = torch.cat([pose_params_start, pose_params_rest]).unsqueeze(0).repeat(batch_size, 1)
    ##print(f"pose_params shape: {pose_params.shape}")

    shape_params = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    ##print(f"shape_params shape: {shape_params.shape}")

    obj_trans = torch.tensor(transformed_trans, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    ##print(f"obj_trans shape: {obj_trans.shape}")

    # GPU mode
    cuda = torch.cuda.is_available()
    ##print(f"CUDA available: {cuda}")
    # device = torch.device("cuda:0" if cuda else "cpu")
    device = torch.device("cpu")
    ##print(f"Device: {device}")

    pose_params = pose_params.to(device)
    shape_params = shape_params.to(device)
    obj_trans = obj_trans.to(device)
    smpl_layer = smpl_layer.to(device)
    ##print("All tensors and models moved to device.")

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
    selected_joints = [
        pelvis,
        left_hip,
        right_hip,
        spine1,
        left_knee,
        right_knee,
        spine2,
        left_ankle,
        right_ankle,
        spine3,
        left_foot,
        right_foot,
        neck,
        left_collar,
        right_collar,
        head,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hand,
        right_hand,
    ]

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
    axis = np.array(
        [
            rot_matrix[2, 1] - rot_matrix[1, 2],
            rot_matrix[0, 2] - rot_matrix[2, 0],
            rot_matrix[1, 0] - rot_matrix[0, 1],
        ]
    )

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


# #breakpoint()
# # Analysis of the distribution
# train_loader = data_module.train_dataloader()

# items = [[], [], [], []]
# means = [[], [], [], []]
# medians = [[], [], [], []]
# std_devs = [[], [], [], []]
# scene = None

# for batch in train_loader:

#     if scene is None:
#         scene = batch[1][0]

#     if scene == batch[1][0]:

#         for item in range(4):
#             for serie in batch[0][item]:
#                 items[item].append(serie.numpy())

#     if scene != batch[1][0]:

#         for item in [2,3]:
#             item_array = np.array(items[item])
#             mean_value = np.mean(item_array, axis=0)
#             median_value = np.median(item_array, axis=0)
#             std_dev = np.std(item_array, axis=0)

#             print(f"Scene {scene}, item {item}, \nMean: \n{mean_value}, \nMedian: \n{median_value}, \nStd Deviation: \n{std_dev}", flush=True)

#             means[item].append(mean_value)
#             medians[item].append(median_value)
#             std_devs[item].append(std_dev)

#         # Reset for the next scene
#         items = [[], [], [], []]
#         scene = batch[1][0]

# for item in [2,3]:
#     means_array = np.array(means)
#     medians_array = np.array(medians)
#     std_devs_array = np.array(std_devs)

#     mean_mean = np.mean(means_array, axis=0)
#     mean_median = np.mean(medians_array, axis=0)
#     mean_std_dev = np.mean(std_devs_array, axis=0)

#     print(f"\n\n\n Overall, item {item}, \nMean: \n{mean_mean}, \nMedian: \n{mean_median}, \nStd Deviation: \n{mean_std_dev}", flush=True)
