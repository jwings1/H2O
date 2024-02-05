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
#     [128, 128, 128, 64, 64, 32, 32]
# ]

# # Number of random configurations to try
# N_TRIALS = 20
# EPOCHS = 300

# learning_rate_range = [1e-5, 1e-4]
# batch_size_range = [64, 128]
# dropout_rate_range = [0.1, 0.2]
# learning_rate_range = [1e-5]
# batch_size_range = [64]
# dropout_rate_range = [0.1]
learning_rate_range = [5e-3]  # , 1e-2, 5e-2]
batch_size_range = [8]
dropout_rate_range = [0.1]
layer_sizes_range = [
    # [128, 128, 64, 64, 32],
    # [256, 256, 128, 128, 64],
    # [512, 512, 256, 256, 128],
    # [1024, 1024, 512, 512, 256],
    # [256, 256, 256, 128, 64, 32],
    # [512, 512, 512, 256, 128, 64],
    # [1024, 1024, 1024, 512, 256, 128],
    # [128, 128, 128, 64, 64, 32, 32]
    # [128, 128, 64, 64, 32],
    # [4096, 4096, 4096, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 512, 256, 128, 64],
    [8192, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
    # [8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64]
    # [8192, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
    # [8192, 8192, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64, 32]
    # [8192, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 512, 256, 128, 64]
]

EPOCHS = 100
best_val_loss = float("inf")
best_params = None

# Grid search over all combinations
for lr, bs, dr, layers in itertools.product(
    learning_rate_range, batch_size_range, dropout_rate_range, layer_sizes_range
):
    LEARNING_RATE = lr
    BATCH_SIZE = bs
    DROPOUT_RATE = dr
    LAYER_SIZES = layers

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

    def load_config(camera_id, base_path, Date):
        config_path = os.path.join(base_path, "calibs", Date, "config", str(camera_id), "config.json")
        with open(config_path, "r") as f:
            return json.load(f)

    def load_split_from_path(path):
        with open(path, "r") as file:
            split_dict = json.load(file)
        return split_dict

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

    # Modified function to load ground truth SMPL with reprojections
    def load_ground_truth_SMPL(ground_path, base_path):
        ground_SMPL_list = []
        reprojected_cam0_list = []
        reprojected_cam2_list = []
        reprojected_cam3_list = []
        paths = []
        identifiers = []
        prev_smpl_data = None

        for filename in glob.iglob(
            os.path.join(ground_path, "**", "person", "fit02", "person_fit.pkl"), recursive=True
        ):
            paths.append(filename)

        paths = sorted(paths)

        for filename in paths:
            with open(filename, "rb") as file:
                data = pickle.load(file)

            Date = filename.split("/")[6].split("_")[0]
            scene = filename.split("/")[6].split("_")[2:]

            if scene == scene:
                smpl_data = np.concatenate([data["pose"][:72], data["trans"]])

                if prev_smpl_data is not None:
                    offset_pose = smpl_data[:72] - prev_smpl_data[:72]
                    offset_trans = smpl_data[-3:] - prev_smpl_data[-3:]
                    offsets = np.concatenate([offset_pose, offset_trans])
                    prev_smpl_data = smpl_data
                else:
                    # Same-initialization
                    offsets = smpl_data
                    prev_smpl_data = smpl_data

                    # Zero-initialization
                    # offsets = np.zeros((75))
                    # prev_smpl_data = np.zeros((75))

                # object_data_1_list.append(offsets)
                ground_SMPL_list.append(offsets)

                pose = offsets[:72]
                trans = offsets[-3:]

                # Reproject to cameras 0, 2, and 3
                camera1_params = load_config(1, base_path, Date)

                # For Camera 0
                cam0_params = load_config(0, base_path, Date)
                reprojected_cam0 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam0_params)
                reprojected_cam0_list.append(reprojected_cam0.flatten())

                # For Camera 2
                cam2_params = load_config(2, base_path, Date)
                reprojected_cam2 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam2_params)
                reprojected_cam2_list.append(reprojected_cam2.flatten())

                # For Camera 3
                cam3_params = load_config(3, base_path, Date)
                reprojected_cam3 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam3_params)
                reprojected_cam3_list.append(reprojected_cam3.flatten())

                identifier = filename.split("/")[6]
                identifiers.append(identifier)

            else:
                prev_object_data = None

                smpl_data = np.concatenate([data["pose"][:72], data["trans"]])

                if prev_smpl_data is not None:
                    offset_pose = smpl_data[:72] - prev_smpl_data[:72]
                    offset_trans = smpl_data[-3:] - prev_smpl_data[-3:]
                    offsets = np.concatenate([offset_pose, offset_trans])
                    prev_smpl_data = smpl_data
                else:
                    # Same-initialization
                    offsets = smpl_data
                    prev_smpl_data = smpl_data

                    # Zero-initialization
                    # offsets = np.zeros((75))
                    # prev_smpl_data = np.zeros((75))

                # object_data_1_list.append(offsets)
                ground_SMPL_list.append(offsets)

                pose = offsets[:72]
                trans = offsets[-3:]

                # Reproject to cameras 0, 2, and 3
                camera1_params = load_config(1, base_path, Date)

                # For Camera 0
                cam0_params = load_config(0, base_path, Date)
                reprojected_cam0 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam0_params)
                reprojected_cam0_list.append(reprojected_cam0.flatten())

                # For Camera 2
                cam2_params = load_config(2, base_path, Date)
                reprojected_cam2 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam2_params)
                reprojected_cam2_list.append(reprojected_cam2.flatten())

                # For Camera 3
                cam3_params = load_config(3, base_path, Date)
                reprojected_cam3 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam3_params)
                reprojected_cam3_list.append(reprojected_cam3.flatten())

                identifier = filename.split("/")[6]
                identifiers.append(identifier)

        return ground_SMPL_list, reprojected_cam0_list, reprojected_cam2_list, reprojected_cam3_list, identifiers

    def load_object_data(object_path, base_path):
        object_data_list = []
        reprojected_cam0_list = []
        reprojected_cam2_list = []
        reprojected_cam3_list = []
        paths = []
        identifiers = []
        object_data_1_list = []
        object_offsets = []
        prev_object_data = None

        for filename in glob.iglob(os.path.join(object_path, "**", "*", "fit01", "*_fit.pkl"), recursive=True):
            paths.append(filename)

        paths = sorted(paths)
        for filename in paths:
            with open(filename, "rb") as file:
                data = pickle.load(file)

            Date = filename.split("/")[6].split("_")[0]
            scene = filename.split("/")[6].split("_")[2:]

            if scene == scene:
                object_data = np.concatenate([data["angle"], data["trans"]])
                object_data_list.append(object_data)

                if prev_object_data is not None:
                    offset_angles = object_data[:3] - prev_object_data[:3]
                    offset_trans = object_data[-3:] - prev_object_data[-3:]
                    offsets = np.concatenate([offset_angles, offset_trans])
                    prev_object_data = object_data
                else:
                    # Same-initialization
                    offsets = object_data
                    prev_object_data = object_data

                    # Zero-initialization
                    # offsets = np.zeros((6))
                    # prev_object_data = np.zeros((6))

                object_data_1_list.append(offsets)

                # Reproject to cameras 0, 2, and 3
                camera1_params = load_config(1, base_path, Date)

                # For Camera 0
                cam0_params = load_config(0, base_path, Date)
                reprojected_cam0 = transform_object_to_camera_frame(offsets, camera1_params, cam0_params)
                reprojected_cam0_list.append(reprojected_cam0.flatten())

                # For Camera 2
                cam2_params = load_config(2, base_path, Date)
                reprojected_cam2 = transform_object_to_camera_frame(offsets, camera1_params, cam2_params)
                reprojected_cam2_list.append(reprojected_cam2.flatten())

                # For Camera 3
                cam3_params = load_config(3, base_path, Date)
                reprojected_cam3 = transform_object_to_camera_frame(offsets, camera1_params, cam3_params)
                reprojected_cam3_list.append(reprojected_cam3.flatten())

                identifier = filename.split("/")[6]
                identifiers.append(identifier)

            else:
                prev_object_data = None

                object_data = np.concatenate([data["angle"], data["trans"]])
                object_data_list.append(object_data)

                if prev_object_data is not None:
                    offset_angles = object_data[:3] - prev_object_data[:3]
                    offset_trans = object_data[-3:] - prev_object_data[-3:]
                    offsets = np.concatenate([offset_angles, offset_trans])
                    prev_object_data = object_data
                else:
                    # Same-initialization
                    offsets = object_data
                    prev_object_data = object_data

                    # Zero-initialization
                    # offsets = np.zeros((6))
                    # prev_object_data = np.zeros((6))

                object_data_1_list.append(offsets)

                # Reproject to cameras 0, 2, and 3
                camera1_params = load_config(1, base_path, Date)

                # For Camera 0
                cam0_params = load_config(0, base_path, Date)
                reprojected_cam0 = transform_object_to_camera_frame(offsets, camera1_params, cam0_params)
                reprojected_cam0_list.append(reprojected_cam0.flatten())

                # For Camera 2
                cam2_params = load_config(2, base_path, Date)
                reprojected_cam2 = transform_object_to_camera_frame(offsets, camera1_params, cam2_params)
                reprojected_cam2_list.append(reprojected_cam2.flatten())

                # For Camera 3
                cam3_params = load_config(3, base_path, Date)
                reprojected_cam3 = transform_object_to_camera_frame(offsets, camera1_params, cam3_params)
                reprojected_cam3_list.append(reprojected_cam3.flatten())

                identifier = filename.split("/")[6]
                identifiers.append(identifier)

        return object_data_1_list, reprojected_cam0_list, reprojected_cam2_list, reprojected_cam3_list, identifiers

    # Update to the BehaveDataset class
    class BehaveDataset(Dataset):
        def __init__(
            self,
            smpl_inputs,
            smpl_reprojected_cam0,
            smpl_reprojected_cam2,
            smpl_reprojected_cam3,
            obj_labels,
            obj_reprojected_cam0,
            obj_reprojected_cam2,
            obj_reprojected_cam3,
            identifiers,
        ):
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
                self.identifiers[idx],
            )

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
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

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
                    init.kaiming_normal_(m.weight)
                    init.zeros_(m.bias)

            # Initialize validation_losses
            self.validation_losses = []

        def forward(self, x):
            for i, linear in enumerate(self.linears[:-1]):  # Exclude the last linear layer
                x = linear(x)
                if i < len(self.bns):  # Apply batch normalization only if available
                    x = self.bns[i](x)
                x = self.dropout(F.relu(x))

            # Last linear layer without activation
            x = self.linears[-1](x)

            return x

        def training_step(self, batch, batch_idx):
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
            wandb.log({"loss_train": avg_loss.item()})  # , step=self.current_epoch)
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
            wandb.log({"loss_val": avg_loss.item()})  # , step=self.current_epoch)
            return {"val_loss": avg_loss}

        def on_validation_epoch_end(self):
            avg_val_loss = torch.mean(torch.tensor(self.validation_losses))
            self.log("loss_val", avg_val_loss, prog_bar=True, logger=True)
            wandb.log({"avg_loss_val": avg_val_loss.item()})  # , step=self.current_epoch)
            self.log_scheduler_info(avg_val_loss.item())
            self.validation_losses = []  # reset for the next epoch

        def log_scheduler_info(self, val_loss):
            scheduler = self.lr_schedulers()
            if isinstance(scheduler, list):
                scheduler = scheduler[0]

            # Log learning rate of the optimizer
            for idx, param_group in enumerate(self.optimizers().param_groups):
                wandb.log({f"learning_rate_{idx}": param_group["lr"]})

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
            optimizer = torch.optim.Adam(
                self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4
            )
            scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True, factor=0.5),
                "monitor": "loss_val",
                "interval": "epoch",
                "frequency": 1,
            }
            return [optimizer], [scheduler]

    # 4. Training using PyTorch Lightnings

    # Integrating the loading and dataset creation

    behave_seq = "/scratch_net/biwidl307_second/lgermano/behave/sequences/Date01_Sub01_backpack_back"
    base_path = "/scratch_net/biwidl307_second/lgermano/behave"
    (
        ground_SMPL_list,
        reprojected_smpl_cam0,
        reprojected_smpl_cam2,
        reprojected_smpl_cam3,
        ground_SMPL_identifiers,
    ) = load_ground_truth_SMPL(behave_seq, base_path)
    (
        object_data_list,
        reprojected_obj_cam0,
        reprojected_obj_cam2,
        reprojected_obj_cam3,
        object_data_identifiers,
    ) = load_object_data(behave_seq, base_path)

    # only predict trans
    object_data_list = [arr[-3:] for arr in object_data_list]
    reprojected_obj_cam0 = [arr[-3:] for arr in reprojected_obj_cam0]
    reprojected_obj_cam2 = [arr[-3:] for arr in reprojected_obj_cam2]
    reprojected_obj_cam3 = [arr[-3:] for arr in reprojected_obj_cam3]

    # Ensure the identifiers from both lists match
    assert ground_SMPL_identifiers == object_data_identifiers

    input_dim = ground_SMPL_list[0].shape[0]
    output_dim = object_data_list[0].shape[0]
    print(input_dim)
    print(output_dim)
    # dataset = BehaveDataset(ground_SMPL_list, reprojected_smpl_cam0, reprojected_smpl_cam2, reprojected_smpl_cam3, object_data_list, reprojected_obj_cam0, reprojected_obj_cam2, reprojected_obj_cam3, object_data_identifiers)
    dataset = BehaveDataset(
        ground_SMPL_list,
        ground_SMPL_list,
        ground_SMPL_list,
        ground_SMPL_list,
        object_data_list,
        object_data_list,
        object_data_list,
        object_data_list,
        object_data_identifiers,
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

    model = MLP(input_dim, output_dim)
    trainer = pl.Trainer(max_epochs=wandb.config.epochs)
    trainer.fit(model, datamodule=data_module)

    # Adjusted computation for average validation loss
    # TO BE FIXED
    if model.validation_losses:
        avg_val_loss = torch.mean(torch.stack(model.validation_losses)).item()
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

    # Optionally, to test the model:
    trainer.test(model, datamodule=data_module)

    # Save the model using WandB run ID
    filename = f"/scratch_net/biwidl307/lgermano/H2O/trained_models/model_offset_trans_{wandb.run.name}.pt"

    # Save the model
    torch.save(model, filename)

    # Finish the current W&B run
    wandb.finish()

# After all trials, print the best set of hyperparameters
print("Best Validation Loss:", best_val_loss)
print("Best Hyperparameters:", best_params)
