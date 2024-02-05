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
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint


EPOCHS = 50
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

learning_rate_range = [5e-4]
# learning_rate_range = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
batch_size_range = [16]
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

    # trainer = Trainer(log_every_n_steps=BATCH_SIZE)  # Log every n steps

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

    def interpolate_frames(all_data_frames):
        interpolated_frames = []

        for idx in range(len(all_data_frames) - 1):
            frame1 = all_data_frames[idx]
            frame2 = all_data_frames[idx + 1]

            # Original frame
            interpolated_frames.append(frame1)

            # Interpolated frames

            for i in range(1, 3):
                interpolated_frame = frame1.copy()
                t = i / 3.0  # Assuming you want to interpolate at 1/3 and 2/3 positions between frame1 and frame2
                interpolated_frame["pose"] = slerp_rotations(frame1["pose"], frame2["pose"], t)
                interpolated_frame["trans"] = linear_interpolate(frame1["trans"], frame2["trans"], t)
                interpolated_frame["obj_pose"] = slerp_rotations(frame1["obj_pose"], frame2["obj_pose"], t)
                interpolated_frame["obj_trans"] = linear_interpolate(frame1["obj_trans"], frame2["obj_trans"], t)

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

    def transform_object_to_camera_frame(data, camera1_params, cam_params):
        """Transform object's position and orientation to another camera frame using relative transformation."""
        # Convert the axis-angle rotation to a matrix

        R_w = Rotation.from_rotvec(data["angle"]).as_matrix()

        # Build transformation matrix of mesh in world coordinates
        T_mesh = np.eye(4)
        T_mesh[:3, :3] = R_w
        T_mesh[:3, 3] = data["trans"]

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

        return transformed_trans

    def render_smpl(transformed_pose, transformed_trans, betas):
        print("Start of render_smpl function.")

        batch_size = 1
        print(f"batch_size: {batch_size}")

        # Create the SMPL layer
        smpl_layer = SMPL_Layer(
            center_idx=0,
            gender="male",
            model_root="/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch/native/models/",
        )
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
        #                 left_foot, right_foot, head, left_shoulder, right_shoulder, left_hand, right_hand]
        return selected_joints

    def project_frames(interpolated_data_frames):
        reprojected_cam1_list = []
        reprojected_cam0_list = []
        reprojected_cam2_list = []
        reprojected_cam3_list = []

        identifiers = []

        base_path = "/scratch_net/biwidl307_second/lgermano/behave"

        # Process interpolated frames
        for idx, frame_data in enumerate(interpolated_data_frames):
            projected_frame = frame_data.copy()

            for cam_id in [1, 0, 2, 3]:
                print(f"\nProcessing for camera {cam_id}...")

                camera1_params = load_config(1, base_path, frame_data["date"])
                cam_params = load_config(cam_id, base_path, frame_data["date"])
                projected_frame["pose"], projected_frame["trans"] = transform_smpl_to_camera_frame(
                    frame_data["pose"], frame_data["trans"], camera1_params, cam_params
                )

                if cam_id == 1:
                    projected_frame["img"] = frame_data["img"][cam_id]

                    # Produce labels: distance joint-obj_trans
                    selected_joints = render_smpl(
                        projected_frame["pose"], projected_frame["trans"], projected_frame["betas"]
                    )
                    selected_joints = [joint.cpu().numpy() for joint in selected_joints]
                    distances = np.asarray(
                        [np.linalg.norm(projected_frame["obj_trans"] - joint) for joint in selected_joints]
                    )
                    projected_frame["distances"] = distances
                    reprojected_cam1_list.append(projected_frame)
                else:
                    if cam_id == 0:
                        projected_frame["img"] = frame_data["img"][cam_id]
                        projected_frame["distances"] = distances
                        reprojected_cam0_list.append(projected_frame)
                    if cam_id == 2:
                        projected_frame["img"] = frame_data["img"][cam_id]
                        projected_frame["distances"] = distances
                        reprojected_cam2_list.append(projected_frame)
                    else:
                        projected_frame["img"] = frame_data["img"][cam_id]
                        projected_frame["distances"] = distances
                        reprojected_cam3_list.append(projected_frame)

                identifiers.append(projected_frame["scene"])

        return reprojected_cam1_list, reprojected_cam0_list, reprojected_cam2_list, reprojected_cam3_list, identifiers

    class BehaveDataset(Dataset):
        def __init__(
            self,
            reprojected_cam0_tensors,
            reprojected_cam1_tensors,
            reprojected_cam2_tensors,
            reprojected_cam3_tensors,
            identifiers,
            num_mlps,
        ):
            self.reprojected_smpl_cam0 = reprojected_cam0_tensors[0]
            self.reprojected_smpl_cam1 = reprojected_cam1_tensors[0]
            self.reprojected_smpl_cam2 = reprojected_cam2_tensors[0]
            self.reprojected_smpl_cam3 = reprojected_cam3_tensors[0]

            self.reprojected_obj_cam0 = reprojected_cam0_tensors[1]
            self.reprojected_obj_cam1 = reprojected_cam0_tensors[1]
            self.reprojected_obj_cam2 = reprojected_cam0_tensors[1]
            self.reprojected_obj_cam3 = reprojected_cam0_tensors[1]

            self.identifiers = identifiers[: len(self.reprojected_smpl_cam0)]
            self.num_mlps = num_mlps

        def __len__(self):
            return len(self.reprojected_smpl_cam0)

        def __getitem__(self, idx):
            return {
                "smpl_cam0": self.reprojected_smpl_cam0[idx],
                "smpl_cam1": self.reprojected_smpl_cam1[idx],
                "smpl_cam2": self.reprojected_smpl_cam2[idx],
                "smpl_cam3": self.reprojected_smpl_cam3[idx],
                "obj_cam0": self.reprojected_obj_cam0[idx],
                "obj_cam1": self.reprojected_obj_cam1[idx],
                "obj_cam2": self.reprojected_obj_cam2[idx],
                "obj_cam3": self.reprojected_obj_cam3[idx],
                "identifier": self.identifiers[idx],
            }

    class BehaveDataModule(pl.LightningDataModule):
        def __init__(self, dataset, split, batch_size=wandb.config.batch_size):
            super(BehaveDataModule, self).__init__()
            self.dataset = dataset
            self.batch_size = batch_size
            self.split = split

            self.train_indices = []
            self.test_indices = []

            for idx, identifier in enumerate(self.dataset.identifiers):
                if identifier in self.split["train"]:
                    self.train_indices.append(idx)
                elif identifier in self.split["test"]:
                    self.test_indices.append(idx)

        def train_dataloader(self):
            train_dataset = Subset(self.dataset, self.train_indices)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)  # , num_workers=16)

        def val_dataloader(self):
            # Assuming validation set is not provided, so using test set as validation
            test_dataset = Subset(self.dataset, self.test_indices)
            return DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
            )  # , num_workers=16)

        def test_dataloader(self):
            test_dataset = Subset(self.dataset, self.test_indices)
            return DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True
            )  # , num_workers=16)

    class MLP(pl.LightningModule):
        def __init__(self, input_dim, output_dim):
            super(MLP, self).__init__()

            self.automatic_optimization = True

            # Use layer_sizes from wandb.config to create the architecture
            layer_sizes = [input_dim] + wandb.config.layer_sizes + [output_dim]
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
            )

            # Batch normalization layers based on the layer sizes
            self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(size) for size in wandb.config.layer_sizes])

            # Dropout layer
            self.dropout = torch.nn.Dropout(wandb.config.dropout_rate)

            # He initialization
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight)
                    torch.nn.init.zeros_(m.bias)

        def forward(self, x):
            for i, linear in enumerate(self.linears[:-1]):  # Exclude the last linear layer
                x = linear(x)
                if i < len(self.bns):  # Apply batch normalization only if available
                    x = self.bns[i](x)
                x = self.dropout(F.relu(x))

            # Last linear layer without activation
            x = self.linears[-1](x)
            return x

    class RecursiveMLP(pl.LightningModule):
        def __init__(self, input_dim_smpl, input_dim_obj, output_dim, num_mlps, initial_obj_pred):
            super(RecursiveMLP, self).__init__()

            self.automatic_optimization = False
            self.num_mlps = num_mlps
            self.initial_obj_pred = initial_obj_pred
            self.criterion = nn.MSELoss()
            # Initialize validation_losses
            self.validation_losses = []

            # Dynamically create the MLP models
            self.mlps = torch.nn.ModuleList()
            self.mlps.append(MLP(input_dim_smpl + input_dim_obj, output_dim))
            for _ in range(1, num_mlps):
                self.mlps.append(MLP(input_dim_smpl + output_dim, output_dim))

        def forward(self, smpl, obj):
            print(smpl.shape)
            print(obj.shape)
            print(type(smpl))
            print(type(obj))
            for i in range(self.num_mlps):
                concatenated_input = torch.cat([smpl, obj], dim=-1)
                current_obj_pred = self.mlps[i](concatenated_input)

            return current_obj_pred

        def training_step(self, batch, batch_idx):
            combined_loss = self._compute_loss(batch)
            wandb.log({"train_loss": combined_loss})
            self.manual_backward(combined_loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()
            return combined_loss

        def validation_step(self, batch, batch_idx):
            combined_loss = self._compute_loss(batch)
            wandb.log({"loss_val": combined_loss})  # Logging loss_val for the scheduler

            self.validation_losses.append(combined_loss)
            return {"val_loss": combined_loss}

        def on_validation_epoch_end(self):
            avg_loss = torch.mean(torch.tensor(self.validation_losses))
            self.log("avg_loss", avg_loss)
            wandb.log({"avg_loss": avg_loss})
            self.validation_losses = []  # reset for the next epoch

        def test_step(self, batch, batch_idx):
            # The structure is the same as the validation step
            outputs = self.validation_step(batch, batch_idx)
            return {"test_loss": outputs["val_loss"]}

        def _compute_loss(self, batch):
            print(batch["smpl_cam0"].shape)

            # Extract the first element for each camera view
            smpl_original_first = batch["smpl_cam0"][:, 0, :]
            smpl_cam0_first = batch["smpl_cam1"][:, 0, :]
            smpl_cam2_first = batch["smpl_cam2"][:, 0, :]
            smpl_cam3_first = batch["smpl_cam3"][:, 0, :]

            obj_original_first = batch["obj_cam0"][:, 0, :]
            obj_cam0_first = batch["obj_cam1"][:, 0, :]
            obj_cam2_first = batch["obj_cam2"][:, 0, :]
            obj_cam3_first = batch["obj_cam3"][:, 0, :]

            obj_original_last = batch["obj_cam0"][:, -1, :]
            obj_cam0_last = batch["obj_cam1"][:, -1, :]
            obj_cam2_last = batch["obj_cam2"][:, -1, :]
            obj_cam3_last = batch["obj_cam3"][:, -1, :]

            # Get the model's predictions using the forward method
            pred_original = self.forward(smpl_original_first, obj_original_first)
            pred_cam0 = self.forward(smpl_cam0_first, obj_cam0_first)
            pred_cam2 = self.forward(smpl_cam2_first, obj_cam2_first)
            pred_cam3 = self.forward(smpl_cam3_first, obj_cam3_first)

            # Compute MSE loss for each camera view
            loss_original = self.criterion(pred_original, obj_original_last)
            loss_cam0 = self.criterion(pred_cam0, obj_cam0_last)
            loss_cam2 = self.criterion(pred_cam2, obj_cam2_last)
            loss_cam3 = self.criterion(pred_cam3, obj_cam3_last)

            # Average the losses from all views
            combined_loss = (loss_original + loss_cam0 + loss_cam2 + loss_cam3) / 4.0

            return combined_loss

        def log_scheduler_info(self, val_loss):
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler = scheduler[0]

            for idx, param_group in enumerate(self.optimizers().param_groups):
                wandb.log({f"learning_rate_{idx}": param_group["lr"]})

            best_metric_val = scheduler.best
            wandb.log({"best_val_loss": best_metric_val})

            epochs_since_improvement = scheduler.num_bad_epochs
            wandb.log({"epochs_since_improvement": epochs_since_improvement})

            scheduler.step(val_loss)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=wandb.config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
            )
            scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True, factor=0.5),
                "monitor": "avg_val_loss",  # Changed to avg_val_loss
                "interval": "step",  # "epoch",
                "frequency": 1,
            }
            return [optimizer], [scheduler]

    # 4. Training using PyTorch Lightnings
    # Integrating the loading and dataset creation
    behave_seq = "/scratch_net/biwidl307_second/lgermano/behave/sequences"
    base_path = "/scratch_net/biwidl307_second/lgermano/behave"
    # all_files = sorted(glob.glob(os.path.join(base_path, "sequences", "**", "t*.000")))

    # ############## POSSIBLY USING A SUBSET ######################
    # selected_files = all_files

    # print(f"Detected {len(selected_files)} frames.")

    # all_data_frames = []

    # # Gather data into all_data_frames
    # for idx, frame_folder in enumerate(selected_files):
    #     frame_data = {}

    #     def get_frame_data(frame_folder):
    #         # For object path
    #         obj_pattern = os.path.join(frame_folder, "*", "fit01", "*_fit.pkl")
    #         obj_matches = glob.glob(obj_pattern)

    #         if obj_matches:
    #             frame_data['obj_path']= obj_matches[0]
    #             #print(frame_data['obj_path'])
    #         else:
    #             obj_path = None

    #         # For SMPL path
    #         smpl_path = os.path.join(frame_folder, "person", "fit02", "person_fit.pkl")
    #         frame_data['smpl_path'] = smpl_path

    #         frame_data['scene'] = frame_folder.split('/')[-2]
    #         frame_data['date'] = frame_folder.split('/')[-2].split('_')[0]
    #         print(frame_data['scene'])
    #         print(frame_data['date'])

    #         return None

    #     get_frame_data(frame_folder)

    #     smpl_data = load_pickle(frame_data['smpl_path'])
    #     frame_data['pose'] = smpl_data['pose'][:72]
    #     frame_data['trans'] = smpl_data['trans']
    #     frame_data['betas'] = smpl_data['betas']

    #     obj_data = load_pickle(frame_data['obj_path'])
    #     frame_data['obj_pose'] = obj_data['angle']
    #     frame_data['obj_trans'] = obj_data['trans']

    #     image_paths = {
    #         1: os.path.join(frame_folder, "k1.color.jpg"),
    #         2: os.path.join(frame_folder, "k2.color.jpg"),
    #         0: os.path.join(frame_folder, "k0.color.jpg"),
    #         3: os.path.join(frame_folder, "k3.color.jpg")
    #     }

    #     frame_data['img'] = image_paths

    #     all_data_frames.append(frame_data)

    # # Interpolate between frames
    # interpolated_data_frames = interpolate_frames(all_data_frames)

    # reprojected_cam1_list, reprojected_cam0_list, reprojected_cam2_list, reprojected_cam3_list, identifiers = project_frames(interpolated_data_frames)

    # data_to_store = {
    #     'reprojected_cam1': reprojected_cam1_list,
    #     'reprojected_cam0': reprojected_cam0_list,
    #     'reprojected_cam2': reprojected_cam2_list,
    #     'reprojected_cam3': reprojected_cam3_list,
    #     'identifiers': identifiers
    # }

    # with open('/scratch_net/biwidl307/lgermano/H2O/datasets/BEHAVE_train_test_int30fps_4cam/dataset_processed.pkl', 'wb') as f:
    #     pickle.dump(data_to_store, f)

    with open(
        "/scratch_net/biwidl307/lgermano/H2O/datasets/BEHAVE_train_test_int30fps_4cam/dataset_processed.pkl", "rb"
    ) as f:
        data_retrieved = pickle.load(f)

    # data_retrieved = data_to_store

    def sliding_window_grouping(input_list, N_MLP, identifiers, group_identifiers):
        """
        Group the input list using overlapping sliding windows with offset 1 and amplitude N_MLP.

        Args:
        - input_list (list): The input list of data.
        - N_MLP (int): The amplitude of the sliding window.

        Returns:
        - list of torch tensors: List containing grouped windows converted to torch tensors.
        """
        tensor_smpl_list = []
        tensor_obj_list = []
        smpl_list = []
        obj_list = []

        for frame in input_list:
            smpl_list.append(np.concatenate([frame["pose"], frame["trans"]]))
            obj_list.append(frame["distances"])

        for i in range(len(input_list) - N_MLP + 1):
            if identifiers[i] == identifiers[i + N_MLP]:
                window_smpl = smpl_list[i : i + N_MLP]
                tensor_smpl = torch.tensor(window_smpl, dtype=torch.float32)
                tensor_smpl_list.append(tensor_smpl)

                window_obj = obj_list[i : i + N_MLP]
                tensor_obj = torch.tensor(window_obj, dtype=torch.float32)
                tensor_obj_list.append(tensor_obj)

                group_identifiers.append(identifiers[i])
            else:
                foo = 1
        return [tensor_smpl_list, tensor_obj_list, group_identifiers]

    group_identifiers = []
    # print(identifiers)

    reprojected_cam1_tensors = sliding_window_grouping(
        data_retrieved["reprojected_cam1"], N_MLPS, data_retrieved["identifiers"], group_identifiers
    )
    reprojected_cam0_tensors = sliding_window_grouping(
        data_retrieved["reprojected_cam0"], N_MLPS, data_retrieved["identifiers"], group_identifiers
    )
    reprojected_cam2_tensors = sliding_window_grouping(
        data_retrieved["reprojected_cam2"], N_MLPS, data_retrieved["identifiers"], group_identifiers
    )
    reprojected_cam3_tensors = sliding_window_grouping(
        data_retrieved["reprojected_cam3"], N_MLPS, data_retrieved["identifiers"], group_identifiers
    )

    group_identifiers = group_identifiers[: len(reprojected_cam1_tensors) - N_MLPS + 1]

    input_dim = 75  # reprojected_smpl_cam0_list[0].shape[0]
    output_dim = 24  # reprojected_obj_cam0_list[0].shape[0]

    # lists of tensored inputs
    dataset = BehaveDataset(
        reprojected_cam0_tensors,
        reprojected_cam1_tensors,
        reprojected_cam2_tensors,
        reprojected_cam3_tensors,
        reprojected_cam3_tensors[2],
        N_MLPS,
    )
    print(reprojected_cam3_tensors[2])
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

    model = RecursiveMLP(
        input_dim_smpl=input_dim,
        input_dim_obj=output_dim,
        output_dim=output_dim,
        num_mlps=N_MLPS,
        initial_obj_pred=INITIAL_OBJ_PRED,
    )

    model = model.float()

    checkpoint_callback = ModelCheckpoint(
        dirpath="/scratch_net/biwidl307/lgermano/H2O/checkpoints",  # directory where the checkpoints will be saved
        filename="{epoch:02d}-{val_loss:.2f}",  # checkpoint file name pattern
        save_top_k=1,  # save only the best checkpoint based on the validation loss
        verbose=True,  # log information
        monitor="avg_loss",  # what metric to monitor (should be in returned log dict from validation_step)
        mode="min",  # 'min' means the checkpoint will be updated when val_loss decreases
        save_last=True,  # always save the last model
    )

    trainer = pl.Trainer(max_epochs=wandb.config.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=data_module)

    # Adjusted computation for average validation loss
    if getattr(model, "val_outputs", []):
        avg_val_loss = torch.stack(model.val_outputs).mean().item()
    else:
        avg_val_loss = float("inf")

    # If current validation loss is the best, update best loss and best params
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_params = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "dropout_rate": DROPOUT_RATE,
            "layer_sizes": LAYER_SIZES,
        }
        wandb.log({"best_val_loss": best_val_loss})

    # Optionally, to test the model:
    trainer.test(model, datamodule=data_module)

    # Save the model using WandB run ID
    filename = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_{wandb.run.name}.pt"

    # Save the model
    torch.save(model, filename)

    # Finish the current W&B run
    wandb.finish()

# After all trials, print the best set of hyperparameters
print("Best Validation Loss:", best_val_loss)
print("Best Hyperparameters:", best_params)
