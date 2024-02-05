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


# Declaring hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 150
DROPOUT_RATE = 0.5
LAYER_SIZES = [256, 256, 128, 128, 64]

# Initializing wandb with hyperparameters
wandb.init(
    project="MLP",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "MLP",
        "dataset": "BEHAVE",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "layer_sizes": LAYER_SIZES,
    },
)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_config(camera_id, base_path):
    config_path = os.path.join(base_path, "calibs", "Date01", "config", str(camera_id), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params):
    """Transform SMPL parameters to another camera frame."""
    global_orientation = pose[:3]
    rot_matrix_global = Rotation.from_rotvec(global_orientation).as_matrix()
    human_position_world = rot_matrix_global @ np.array([0, 0, 0]).reshape(3, 1) + trans.reshape(3, 1)

    rotation1_inv = np.array(camera1_params["rotation"]).reshape(3, 3).T
    translation1_inv = -np.array(camera1_params["translation"]).reshape(3, 1)
    rotation_cam = np.array(cam_params["rotation"]).reshape(3, 3)
    translation_cam = np.array(cam_params["translation"]).reshape(3, 1)

    human_position_cam = rotation_cam @ (rotation1_inv @ human_position_world + translation1_inv) + translation_cam
    return human_position_cam


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


def project_to_image(point_3d, intrinsics, distortion_coeffs):
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    k1, k2, k3 = distortion_coeffs["k1"], distortion_coeffs["k2"], distortion_coeffs["k3"]
    p1, p2 = distortion_coeffs["p1"], distortion_coeffs["p2"]

    x_norm = point_3d[0] / point_3d[2]
    y_norm = point_3d[1] / point_3d[2]

    r2 = x_norm**2 + y_norm**2
    x_distorted = (
        x_norm * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
    )
    y_distorted = (
        y_norm * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) + 2 * p2 * x_norm * y_norm + p1 * (r2 + 2 * y_norm**2)
    )

    u = fx * x_distorted + cx
    v = fy * y_distorted + cy

    return int(u), int(v)


def load_ground_truth_SMPL(ground_path):
    ground_SMPL_list = []
    paths = []
    identifiers = []

    for filename in glob.iglob(os.path.join(ground_path, "**", "person", "fit02", "person_fit.pkl"), recursive=True):
        paths.append(filename)

    paths = sorted(paths)
    for filename in paths:
        with open(filename, "rb") as file:
            data = pickle.load(file)
        SMPL_ground = np.concatenate([data["pose"], data["betas"], data["trans"], [data["score"]]])
        ground_SMPL_list.append(SMPL_ground)

        identifier = filename.split("/")[6]
        identifiers.append(identifier)

    return ground_SMPL_list, identifiers


def load_object_data(object_path):
    object_data_list = []
    paths = []
    identifiers = []

    for filename in glob.iglob(os.path.join(object_path, "**", "*", "fit01", "*_fit.pkl"), recursive=True):
        paths.append(filename)

    paths = sorted(paths)
    for filename in paths:
        with open(filename, "rb") as file:
            data = pickle.load(file)
        object_data = np.concatenate([data["angle"], data["trans"]])
        object_data_list.append(object_data)

        identifier = filename.split("/")[6]
        identifiers.append(identifier)

    return object_data_list, identifiers


def load_split_from_path(path):
    with open(path, "r") as file:
        split_dict = json.load(file)
    return split_dict


class BehaveDataset(Dataset):
    def __init__(self, inputs, labels, identifiers):
        self.inputs = inputs
        self.labels = labels
        self.identifiers = identifiers

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
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

        # Increased layers and neurons
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, output_dim)

        # Dropout layers for regularization
        self.dropout = torch.nn.Dropout(0.5)

        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)

        x = F.relu(self.fc5(x))
        x = self.dropout(x)

        return self.fc6(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        wandb.log({"loss_train": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        wandb.log({"loss_val": loss.item()})
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        wandb.log({"loss_test": loss.item()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=wandb.config.learning_rate)


# 4. Training using PyTorch Lightnings
# Integrate the loading and dataset creation
behave_seq = "/scratch_net/biwidl307_second/lgermano/behave/sequences"
ground_SMPL_list, ground_SMPL_identifiers = load_ground_truth_SMPL(behave_seq)
object_data_list, object_data_identifiers = load_object_data(behave_seq)
input_dim = ground_SMPL_list[0].shape[0]
output_dim = object_data_list[0].shape[0]

# Ensure the identifiers from both lists match
assert ground_SMPL_identifiers == object_data_identifiers

dataset = BehaveDataset(ground_SMPL_list, object_data_list, ground_SMPL_identifiers)

path_to_file = "/scratch_net/biwidl307_second/lgermano/behave/split.json"
split_dict = load_split_from_path(path_to_file)

data_module = BehaveDataModule(dataset, split=split_dict)
model = MLP(input_dim, output_dim)
# If a GPU is available, use it (set gpus=-1 to use all available GPUs)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, datamodule=data_module)

# Optionally, to test the model:
trainer.test(model, datamodule=data_module)

# 5. Save and log the model
model_filename = "MLP_model.pt"  # specify a filename
model_save_path = os.path.join("/scratch_net/biwidl307/lgermano/crossvit/trained_models", model_filename)
torch.save(model.state_dict(), model_save_path)

# Log the model to W&B
wandb.save(model_save_path)

# Finish the W&B run
wandb.finish()
