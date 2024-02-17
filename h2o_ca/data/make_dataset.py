import os
import pickle
import numpy as np
import gc
from data.labels import labels
from data.behave_dataset import BehaveDatasetOffset, BehaveDataModule
from data.utils import *

# Parameters
frames_subclip = wandb.config.frames_subclip
masked_frames = wandb.config.masked_frames
l = wandb.config.L
create_new_dataset = wandb.config.create_new_dataset
load_existing_dataset = wandb.config.load_existing_dataset
save_data_module = wandb.config.save_data_module
load_data_module = wandb.config.load_data_module

# breakpoint()

# Paths and settings
data_file_path = "/scratch_net/biwidl307/lgermano/H2O/datasets/behave_test8.pkl"
base_path_annotations = "/scratch_net/biwidl307_second/lgermano/behave/behave-30fps-params-v1"
base_path_trace = "/srv/beegfs02/scratch/3dhumanobjint/data/TRACE_results"
base_path_template = "/scratch_net/biwidl307_second/lgermano/behave"
path_to_file = "/scratch_net/biwidl307_second/lgermano/behave/split.json"

cam_ids = wandb.config.cam_ids  # [1]
# E.g. selected_keys = ["SMPL_pose", "SMPL_joints", "OBJ_pose", "OBJ_trans"]
selected_keys = [
    wandb.config.first_option,
    wandb.config.second_option,
    wandb.config.third_option,
    wandb.config.fourth_option,
]
split_dict = load_split_from_path(path_to_file)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create or load dataset
dataset = None
if load_existing_dataset:
    if os.path.exists(data_file_path):
        # Load the saved data
        with open(data_file_path, "rb") as f:
            dataset = pickle.load(f)

if create_new_dataset:
    # Create a new dataset

    processed_path = "/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy"
    print("Processing labels:", labels, flush=True)

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

            all_data_frames_int = interpolate_frames(all_data_frames, l)
            print("Interpolation done. Length of interpolated frames:", len(all_data_frames_int), flush=True)
            del all_data_frames

            objects = project_frames(all_data_frames_int, timestamps, l, base_path_template, base_path_trace, label)
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

# Initialize BehaveDataModule and save or load
data_module = None
if save_data_module:
    # Contains all the logic
    behave_dataset = BehaveDatasetOffset(labels, cam_ids, frames_subclip, selected_keys, wandb, device)

    # Combine wandb.run.name to create a unique name for the saved file
    save_file_name = f"test.pt"
    data_file_path = "/srv/beegfs02/scratch/3dhumanobjint/data/H2O/data_module"
    full_save_path = os.path.join(data_file_path, save_file_name)

    data_module = BehaveDataModule(behave_dataset, split_dict, len(behave_dataset.data_info))

    # Save the data module locally
    with open(full_save_path, "wb") as f:
        pickle.dump(data_module, f)

if load_data_module:
    save_file_name = f"test.pt"
    data_file_path = "/srv/beegfs02/scratch/3dhumanobjint/data/H2O/data_module"
    full_save_path = os.path.join(data_file_path, save_file_name)

    # Load the data
    with open(full_save_path, "rb") as f:
        data_module = pickle.load(f)

print("Dataset loaded", flush=True)
