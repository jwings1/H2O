import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation
import torch.nn as nn
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import open3d as o3d
from labels import labels
from behave_dataset import BehaveDatasetOffset, BehaveDataModule
from utils import *

# Options for creating/loading dataset and data module
create_new_dataset = False  # Set to True to create a new dataset
load_existing_dataset = True  # Set to True to load an existing dataset
save_data_module = True  # Set to True to save a data module
load_data_module = False  # Set to True to load a data module

# Retrieve dataset
data_file_path = "/scratch_net/biwidl307/lgermano/H2O/datasets/behave_test8.pkl"

# Dataset creation
base_path_annotations = "/scratch_net/biwidl307_second/lgermano/behave/behave-30fps-params-v1"
base_path_trace = "/srv/beegfs02/scratch/3dhumanobjint/data/TRACE_results"
base_path_template = "/scratch_net/biwidl307_second/lgermano/behave"
path_to_file = "/scratch_net/biwidl307_second/lgermano/behave/split.json"

# Define labels, camera IDs, and frame range
cam_ids = [0, 1, 2, 3]
frames_subclip = 12
masked_frames = 4
selected_keys = ["SMPL_pose", "SMPL_joints", "OBJ_pose", "OBJ_trans"]  # Add other keys as needed
split_dict = load_split_from_path(path_to_file)

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to create or load dataset
def create_or_load_dataset():
    if os.path.exists(data_file_path) and not create_new_dataset:
        # Load the saved data
        with open(data_file_path, "rb") as f:
            dataset = pickle.load(f)
    elif create_new_dataset:
        # Create a new dataset
        N = 1

        wandb.run.name = wandb.run.name + name
        processed_path = "/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy"
        print("Processing labels:", labels, flush=True)

        dataset = []

        for label in labels:
            if True:  # label not in labels_existing:
                print("Processing label:", label, flush=True)
                selected_file_path_obj = os.path.join(base_path_annotations, label, "object_fit_all.npz")
                selected_file_path_smpl = os.path.join(base_path_annotations, label, "smpl_fit_all.npz")

                print("Object file path:", selected_file_path_obj, flush=True)
                print("SMPL file path:", selected_file_path_smpl, flush=True)

                all_data_frames = []

                with np.load(selected_file_path_obj, allow_pickle=True) as data_obj:
                    print("Loading object data", flush=True)
                    obj_pose = data_obj["angles"]
                    obj_trans = data_obj["trans"]
                    timestamps = data_obj["frame_times"]
                    print("Object data loaded. Shape:", obj_pose.shape, flush=True)

                with np.load(selected_file_path_smpl, allow_pickle=True) as data_smpl:
                    print("Loading SMPL data", flush=True)
                    pose = data_smpl["poses"][:, :72]
                    trans = data_smpl["trans"]
                    betas = data_smpl["betas"]
                    print("SMPL data loaded. Shape:", pose.shape, flush=True)

                for idx in range(min(trans.shape[0], obj_trans.shape[0])):
                    frame_data = {}
                    obj_name = label.split("_")[2]
                    frame_data["obj_template_path"] = os.path.join(
                        base_path_template, "objects", obj_name, obj_name + ".obj"
                    )
                    frame_data["scene"] = label
                    frame_data["date"] = label.split("_")[0]
                    frame_data["pose"] = pose[idx, :]
                    frame_data["trans"] = trans[idx, :]
                    frame_data["betas"] = betas[idx, :]
                    frame_data["obj_pose"] = obj_pose[idx, :]
                    frame_data["obj_trans"] = obj_trans[idx, :]

                    all_data_frames.append(frame_data)

                all_data_frames_int = interpolate_frames(all_data_frames, N)
                print("Interpolation done. Length of interpolated frames:", len(all_data_frames_int), flush=True)
                del all_data_frames

                objects = project_frames(all_data_frames_int, timestamps, N)
                print("Projection done. Length of projected frames:", len(objects), flush=True)
                del all_data_frames_int

                for j in range(len(objects)):
                    for k in range(len(objects[j])):
                        a = objects[j][k]
                        for key in a.keys():
                            if hasattr(a[key], "numpy"):
                                objects[j][k][key] = a[key].numpy()
                            else:
                                objects[j][k][key] = a[key]

                data_file_path = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy/{label}.pkl"
                print(f"Saving data to {data_file_path}", flush=True)
                with open(data_file_path, "wb") as f:
                    pickle.dump(objects, f)
                print(f"Saved data for {label} to {data_file_path}", flush=True)
                del objects
                gc.collect()
    return dataset

# Create or load dataset
dataset = create_or_load_dataset()

if save_data_module:
    # Contains all the logic
    behave_dataset = BehaveDatasetOffset(labels, cam_ids, frames_subclip, selected_keys, wandb, device)

    # Combine wandb.run.name to create a unique name for the saved file
    save_file_name = f"{wandb.run.name}_BEHAVE_singlebatch.pt"
    data_file_path = "/srv/beegfs02/scratch/3dhumanobjint/data/H2O/data_module"
    full_save_path = os.path.join(data_file_path, save_file_name)

    data_module = BehaveDataModule(
        behave_dataset, split_dict, len(behave_dataset.data_info)
    )

    # Save the data module locally
    with open(full_save_path, "wb") as f:
        pickle.dump(data_module, f)

if load_data_module:
    # Load the data
    with open(full_save_path, 'rb') as f:
        data_module = pickle.load(f)

print("Dataset loaded", flush=True)
