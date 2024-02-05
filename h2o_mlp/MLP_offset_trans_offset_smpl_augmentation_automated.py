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
import scipy.spatial.transform as spt
import torch.nn as nn
import h2o
from h2o.automl import H2OAutoML
from sklearn.decomposition import PCA
import numpy as np
import time

# Initialize H2O
h2o.init()

# Set the WANDB_CACHE_DIR environment variable
os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_config(camera_id, base_path, Date):
    config_path = os.path.join(base_path, "calibs", Date, "config", str(camera_id), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_split_from_path(path):
    with open(path, "r") as file:
        split_dict = json.load(file)
    return split_dict


def geodesic_loss(r1, r2):
    """Compute the geodesic distance between two axis-angle representations."""
    return torch.min(torch.norm(r1 - r2, dim=-1), torch.norm(r1 + r2, dim=-1)).mean()


def transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params):
    # Convert axis-angle representation to rotation matrix
    mesh_rotation = Rotation.from_rotvec(pose[:3]).as_matrix()

    # Build transformation matrix of mesh in world coordinates
    T_mesh = np.eye(4)
    T_mesh[:3, :3] = mesh_rotation
    T_mesh[:3, 3] = trans

    # Extract rotation and translation of camera from world coordinates
    R_w_c = np.array(cam_params["rotation"]).reshape(3, 3)
    t_w_c = np.array(cam_params["translation"]).reshape(3, 1)

    # Build transformation matrix of camera in world coordinates
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_w_c
    T_cam[:3, 3:4] = t_w_c

    # Compute transformation of mesh in camera coordinate frame
    T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh

    # Extract transformed pose and translation of mesh in camera coordinate frame
    transformed_pose = Rotation.from_matrix(T_mesh_in_cam[:3, :3]).as_rotvec().flatten()
    transformed_pose = np.concatenate([transformed_pose, pose[3:]]).flatten()
    transformed_trans = T_mesh_in_cam[:3, 3].flatten()

    return np.concatenate([transformed_pose, transformed_trans])


def transform_object_to_camera_frame(obj_data, camera1_params, cam_params):
    # Extract rotation and translation of camera from world coordinates
    R_cam = np.array(cam_params["rotation"]).reshape(3, 3)
    t_w_c = np.array(cam_params["translation"]).reshape(3, 1)
    R_obj = Rotation.from_rotvec(obj_data[:3]).as_matrix()

    # Compute transformation matrix for object
    T_obj = np.eye(4)
    T_obj[:3, :3] = R_obj
    T_obj[:3, 3] = obj_data[-3:]

    # Compute transformation matrix for camera
    T_cam = np.eye(4)
    T_cam[:3, :3] = R_cam
    T_cam[:3, 3] = t_w_c.flatten()

    # Compute the overall transformation matrix
    T_overall = np.linalg.inv(T_cam) @ T_obj

    # transformed_pose = Rotation.from_matrix(T_overall[:3, :3]).as_rotvec().flatten()
    transformed_trans = T_overall[:3, 3].flatten()

    return np.concatenate([transformed_trans])


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

    prev_smpl_data = None
    Date = "Date01"

    # Process interpolated frames
    for idx, frame_data in enumerate(interpolated_data_frames):
        pose = frame_data["pose"]
        trans = frame_data["trans"]
        obj_pose = frame_data["obj_pose"]
        obj_trans = frame_data["obj_trans"]

        smpl_data = np.concatenate([pose[:72], trans])
        object_data = np.concatenate([obj_pose, obj_trans])

        ###########################
        # NOT PREDICTING OFFSETS add and False
        ###########################

        if prev_smpl_data is not None and False:
            offset_pose_smpl = smpl_data[:72] - prev_smpl_data[:72]
            offset_trans_smpl = smpl_data[-3:] - prev_smpl_data[-3:]
            offsets_smpl = np.concatenate([offset_pose_smpl, offset_trans_smpl])
            prev_smpl_data = smpl_data

            offset_pose_obj = object_data[:3] - prev_object_data[:3]
            offset_trans_obj = object_data[-3:] - prev_object_data[-3:]
            offsets_obj = np.concatenate([offset_pose_obj, offset_trans_obj])
            prev_object_data = object_data
        else:
            # Same-initialization
            offsets_smpl = smpl_data
            prev_smpl_data = smpl_data

            # Same-initialization
            offsets_obj = object_data
            prev_object_data = object_data

            # Zero-initialization
            # offsets = np.zeros((75))
            # prev_smpl_data = np.zeros((75))
            # offsets = np.zeros((6))
            # prev_object_data = np.zeros((6))

        reprojected_smpl_cam1_list.append(offsets_smpl)
        reprojected_obj_cam1_list.append(offsets_obj)

        pose = offsets_smpl[:72]
        trans = offsets_smpl[-3:]
        # pose and trans!
        pose_obj = offsets_obj

        # Reproject to cameras 0, 2, and 3
        camera1_params = load_config(1, base_path, Date)

        # For Camera 0
        cam0_params = load_config(0, base_path, Date)
        reprojected_cam0 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam0_params)
        reprojected_smpl_cam0_list.append(reprojected_cam0.flatten())

        reprojected_cam0_obj = transform_object_to_camera_frame(pose_obj, camera1_params, cam0_params)
        reprojected_obj_cam0_list.append(reprojected_cam0_obj.flatten())

        # For Camera 2
        cam2_params = load_config(2, base_path, Date)
        reprojected_cam2 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam2_params)
        reprojected_smpl_cam2_list.append(reprojected_cam2.flatten())

        reprojected_cam2_obj = transform_object_to_camera_frame(pose_obj, camera1_params, cam2_params)
        reprojected_obj_cam2_list.append(reprojected_cam2_obj.flatten())

        # For Camera 3
        cam3_params = load_config(3, base_path, Date)
        reprojected_cam3 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam3_params)
        reprojected_smpl_cam3_list.append(reprojected_cam3.flatten())

        reprojected_cam3_obj = transform_object_to_camera_frame(pose_obj, camera1_params, cam3_params)
        reprojected_obj_cam3_list.append(reprojected_cam3_obj.flatten())

        # identifier = filename.split('/')[6]
        identifier = "Date01_Sub01_backpack_back"
        identifiers.append(identifier)

    return (
        reprojected_smpl_cam1_list,
        reprojected_smpl_cam0_list,
        reprojected_smpl_cam2_list,
        reprojected_smpl_cam3_list,
        reprojected_obj_cam1_list,
        reprojected_obj_cam0_list,
        reprojected_obj_cam2_list,
        reprojected_obj_cam3_list,
        identifiers,
    )


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


def interpolate_frames(all_data_frames):
    interpolated_frames = []

    for idx in range(len(all_data_frames) - 1):
        frame1 = all_data_frames[idx]
        frame2 = all_data_frames[idx + 1]

        # Original frame
        interpolated_frames.append(frame1)

        # Interpolated frames
        # No interpolation!!!!!
        for i in range(5, 5):
            interpolated_frame = frame1.copy()
            t = i / 5.0  # Assuming you want to interpolate at 1/3 and 2/3 positions between frame1 and frame2
            interpolated_frame["pose"] = slerp_rotations(frame1["pose"], frame2["pose"], t)
            interpolated_frame["trans"] = linear_interpolate(frame1["trans"], frame2["trans"], t)
            interpolated_frame["obj_pose"] = slerp_rotations(frame1["obj_pose"], frame2["obj_pose"], t)
            interpolated_frame["obj_trans"] = linear_interpolate(frame1["obj_trans"], frame2["obj_trans"], t)

            interpolated_frames.append(interpolated_frame)

    # Adding the last original frame
    interpolated_frames.append(all_data_frames[-1])

    return interpolated_frames


# 4. Training using H2O

base_path = "/scratch_net/biwidl307_second/lgermano/behave"
############## USING A SUBSET - BACKPACK UP ######################
all_files = sorted(glob.glob(os.path.join(base_path, "sequences", "Date01_Sub01_backpack_back", "t*.000")))
selected_files = all_files  # [:13] + all_files[19:26] + all_files[34:]


print(f"Detected {len(selected_files)} frames.")

all_data_frames = []

# Gather data into all_data_frames
for idx, frame_folder in enumerate(selected_files):
    frame_data = {}

    frame_data["smpl_path"] = os.path.join(frame_folder, "person", "fit02", "person_fit.pkl")
    object_name = "backpack"
    frame_data["obj_path"] = os.path.join(frame_folder, object_name, "fit01", f"{object_name}_fit.pkl")
    frame_data["scene"] = os.path.basename(frame_folder)

    smpl_data = load_pickle(frame_data["smpl_path"])
    frame_data["pose"] = smpl_data["pose"]
    frame_data["trans"] = smpl_data["trans"]
    frame_data["betas"] = smpl_data["betas"]

    obj_data = load_pickle(frame_data["obj_path"])
    frame_data["obj_pose"] = obj_data["angle"]
    frame_data["obj_trans"] = obj_data["trans"]

    image_paths = {
        1: os.path.join(frame_folder, "k1.color.jpg"),
        2: os.path.join(frame_folder, "k2.color.jpg"),
        0: os.path.join(frame_folder, "k0.color.jpg"),
        3: os.path.join(frame_folder, "k3.color.jpg"),
    }

    frame_data["img"] = image_paths

    all_data_frames.append(frame_data)

# Interpolate frames
interpolated_data_frames = interpolate_frames(all_data_frames)

(
    reprojected_smpl_cam1_list,
    reprojected_smpl_cam0_list,
    reprojected_smpl_cam2_list,
    reprojected_smpl_cam3_list,
    reprojected_obj_cam1_list,
    reprojected_obj_cam0_list,
    reprojected_obj_cam2_list,
    reprojected_obj_cam3_list,
    identifiers,
) = project_frames(interpolated_data_frames)

# #Only predict trans
# reprojected_obj_cam1_list = [arr[-3:] for arr in reprojected_obj_cam1_list]
# reprojected_obj_cam0_list = [arr[-3:] for arr in reprojected_obj_cam0_list]
# reprojected_obj_cam2_list = [arr[-3:] for arr in reprojected_obj_cam2_list]
# reprojected_obj_cam3_list = [arr[-3:] for arr in reprojected_obj_cam3_list]

# Only predict pose
reprojected_obj_cam1_list = [arr[:3] for arr in reprojected_obj_cam1_list]
reprojected_obj_cam0_list = [arr[:3] for arr in reprojected_obj_cam0_list]
reprojected_obj_cam2_list = [arr[:3] for arr in reprojected_obj_cam2_list]
reprojected_obj_cam3_list = [arr[:3] for arr in reprojected_obj_cam3_list]

# From smpl trans
# reprojected_smpl_cam1_list = [arr[-3:] for arr in reprojected_smpl_cam1_list]
# reprojected_smpl_cam0_list = [arr[-3:] for arr in reprojected_smpl_cam0_list]
# reprojected_smpl_cam2_list = [arr[-3:] for arr in reprojected_smpl_cam2_list]
# reprojected_smpl_cam3_list = [arr[-3:] for arr in reprojected_smpl_cam3_list]
# Ensure the identifiers from both lists match
# assert ground_SMPL_identifiers == object_data_identifiers

# input_dim = reprojected_smpl_cam1_list[0].shape[0]
# output_dim = reprojected_obj_cam1_list[0].shape[0]

############## CAM 0-3 ###################################

# Concatenate the input arrays
input_data = np.concatenate(
    [reprojected_smpl_cam1_list, reprojected_smpl_cam0_list, reprojected_smpl_cam2_list, reprojected_smpl_cam3_list],
    axis=0,
)

# Concatenate the label arrays
labels = np.concatenate(
    [reprojected_obj_cam1_list, reprojected_obj_cam0_list, reprojected_obj_cam2_list, reprojected_obj_cam3_list], axis=0
)

# Convert numpy arrays to H2O Frame
data = h2o.H2OFrame(np.hstack((input_data, labels)))

# Assuming columns [0:72] are 'pose_params', [72:75] are 'trans_params', and [75:78] are 'axis_angle_representation'
data.columns = [f"pose_{i}" for i in range(72)] + [
    "trans_x",
    "trans_y",
    "trans_z",
    "obj_pose_x",
    "obj_pose_y",
    "obj_pose_z",
]

# Normalize pose and trans parameters
for col in data.columns[0:72] + data.columns[72:75]:
    data[col] = data[col].scale()

# Dimensionality Reduction on pose parameters using PCA
pose_params_np = h2o.as_list(data[:, 0:72]).values
pca = PCA(n_components=30)  # Assuming we want to reduce to 30 components
pose_pca = pca.fit_transform(pose_params_np)
pose_pca_h2o = h2o.H2OFrame(pose_pca, column_names=[f"pca_{i}" for i in range(30)])

# Replace the first 72 columns with the 30 PCA columns
data = data[:, 72:]  # Drop the original 72 columns
data = pose_pca_h2o.cbind(data)  # Add the 30 PCA columns to the beginning

# Split the data into train and test sets (80% train, 20% test)
train, test = data.split_frame(ratios=[0.8])

# Print size of train/test sets
print(f"Number of rows in train set: {train.nrows}")
print(f"Number of rows in test set: {test.nrows}")

# Define the input features and target labels
x = train.columns.copy()
target_columns = ["obj_pose_x", "obj_pose_y", "obj_pose_z"]
for target in target_columns:
    x.remove(target)
y = target_columns[0]

# Initialize and train the H2O AutoML model
aml = H2OAutoML(max_models=20000, seed=1, max_runtime_secs=14400)
aml.train(x=x, y=y, training_frame=train)

# Display the leaderboard
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

# # Predict on test data
# predictions = aml.predict(test)

# # Convert predictions to list
# predictions_list = h2o.as_list(predictions)['predict'].tolist()

# print("Predictions:", predictions_list)

# Save the leader model
leader_model = aml.leader

# Create a timestamped filename
current_time = time.strftime("%Y%m%d-%H%M%S")
save_path = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_h2o_{current_time}"

# Save the model
h2o.save_model(model=leader_model, path=save_path, force=True)

# Shutdown H2O instance
h2o.shutdown(prompt=False)
