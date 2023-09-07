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
os.environ['WANDB_CACHE_DIR'] = '/scratch_net/biwidl307/lgermano/crossvit/wandb/cache'

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
learning_rate_range = [5e-3]
batch_size_range = [16]
dropout_rate_range = [0.1]
layer_sizes_range = [
    #[128, 128, 64, 64, 32],
    # [256, 256, 128, 128, 64], 
    # [512, 512, 256, 256, 128], 
    # [1024, 1024, 512, 512, 256],
    [256, 256, 256, 128, 64, 32],
    # [512, 512, 512, 256, 128, 64],
    # [1024, 1024, 1024, 512, 256, 128],
    # [128, 128, 128, 64, 64, 32, 32]
]
    #[128, 128, 64, 64, 32],
    #[4096, 4096, 4096, 2048, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 512, 256, 128, 64],
    #[8192, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
    #[8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64]
    #[8192, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64],
    #[8192, 8192, 8192, 8192, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 512, 512, 256, 128, 64, 32]
    #[8192, 8192, 8192, 8192, 8192, 4096, 4096, 4096, 2048, 2048, 2048, 1024, 1024, 1024, 512, 512, 512, 256, 128, 64]



EPOCHS = 1e4
best_val_loss = float('inf')
best_params = None

# Grid search over all combinations
for lr, bs, dr, layers in itertools.product(learning_rate_range, batch_size_range, dropout_rate_range, layer_sizes_range):
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
        }
    )

    def load_pickle(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_config(camera_id, base_path, Date):
        config_path = os.path.join(base_path, 'calibs', Date, 'config', str(camera_id), 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def load_split_from_path(path):
        with open(path, 'r') as file:
            split_dict = json.load(file)
        return split_dict

    def transform_smpl_to_camera_frame(pose, trans, camera1_params, cam_params):
        """Transform SMPL parameters to another camera frame."""
        
        # Extract the global orientation (first three parameters) from the pose
        global_orientation = pose[:3]
        
        # Convert the rotation vector to a rotation matrix
        rot_matrix_global = Rotation.from_rotvec(global_orientation).as_matrix()
        
        # Get the inverse rotation for camera1
        rotation1_inv = np.array(camera1_params['rotation']).reshape(3, 3).T
        
        # Apply the inverse rotation of camera1 to the global orientation
        rot_matrix_transformed = rotation1_inv @ rot_matrix_global

        # Convert back to rotation vector
        transformed_global_orientation = Rotation.from_matrix(rot_matrix_transformed).as_rotvec()
        
        # Get the rotation and translation for the target camera (cam_params)
        rotation_cam = np.array(cam_params['rotation']).reshape(3, 3)
        translation_cam = np.array(cam_params['translation']).reshape(3, 1)

        # Transform the trans to the target camera's frame
        transformed_trans = (rotation_cam @ (rotation1_inv @ trans.reshape(3, 1) + translation_cam)).flatten()
        # Only the first three parameters of the pose are reprojected, the rest are left as they are
        transformed_pose = (np.concatenate([transformed_global_orientation, pose[3:]])).flatten()

        return np.concatenate([transformed_pose, transformed_trans])


    def transform_object_to_camera_frame(obj_pose, camera1_params, cam_params):
        """Transform object's position and orientation to another camera frame using relative transformation."""
        
        # Convert the axis-angle rotation to a matrix
        # rot_matrix_obj = Rotation.from_rotvec(obj_pose['angle']).as_matrix()
        # translation_obj = np.array(obj_pose['trans']).reshape(3, 1)
        
        # Compute object position in world frame
        #obj_position_world = rot_matrix_obj @ np.array([0, 0, 0]).reshape(3, 1) + translation_obj
        #obj_position_world = translation_obj

        # # Get the relative transformation from camera 1 to the current camera
        # rotation1_inv = np.array(camera1_params['rotation']).reshape(3, 3).T
        # translation1_inv = -np.array(camera1_params['translation']).reshape(3, 1)
        # rotation_cam = np.array(cam_params['rotation']).reshape(3, 3)
        # translation_cam = np.array(cam_params['translation']).reshape(3, 1)
        
        # # Apply the relative transformation to the object's position in world frame
        # obj_position_cam = rotation_cam @ (rotation1_inv @ obj_position_world + translation1_inv) + translation_cam
        
        # # Transform the object's orientation
        # rot_matrix_obj_cam_frame = rotation_cam @ rotation1_inv @ rot_matrix_obj
        # obj_angle_cam_frame = Rotation.from_matrix(rot_matrix_obj_cam_frame).as_rotvec()
        
        # return np.concatenate([obj_angle_cam_frame, obj_position_cam.flatten()])

        # # Get the relative transformation from camera 1 to the current camera
        # rotation1_inv = np.array(camera1_params['rotation']).reshape(3, 3).T
        # translation1_inv = -np.array(camera1_params['translation']).reshape(3, 1)
        # rotation_cam = np.array(cam_params['rotation']).reshape(3, 3)
        # translation_cam = np.array(cam_params['translation']).reshape(3, 1)
        
        # # Apply the relative transformation to the object's position in world frame
        # obj_position_cam = rotation_cam @ (rotation1_inv @ obj_position_world + translation1_inv) + translation_cam
        
        # return obj_position_cam.flatten()


        # Convert the axis-angle rotation to a matrix
        rot_matrix_obj = Rotation.from_rotvec(obj_pose['angle']).as_matrix()

        # Get the relative transformation from camera 1 to the current camera
        rotation1_inv = np.array(camera1_params['rotation']).reshape(3, 3).T
        rotation_cam = np.array(cam_params['rotation']).reshape(3, 3)

        # Transform the object's orientation
        rot_matrix_obj_cam_frame = rotation_cam @ rotation1_inv @ rot_matrix_obj
        obj_angle_cam_frame = Rotation.from_matrix(rot_matrix_obj_cam_frame).as_rotvec()
        
        return obj_angle_cam_frame.flatten()

    # Modified function to load ground truth SMPL with reprojections
    def load_ground_truth_SMPL(ground_path, base_path):
        ground_SMPL_list = []
        reprojected_cam0_list = []
        reprojected_cam2_list = []
        reprojected_cam3_list = []
        paths = []
        identifiers = []

        for filename in glob.iglob(os.path.join(ground_path, '**', 'person', 'fit02', 'person_fit.pkl'), recursive=True):
            paths.append(filename)

        paths = sorted(paths)
        
        for filename in paths:
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            SMPL_ground = np.concatenate([data['pose'], data['trans']])
            ground_SMPL_list.append(SMPL_ground)

            Date = filename.split('/')[6].split('_')[0]

            # Reproject to cameras 0, 2, and 3
            camera1_params = load_config(1, base_path, Date)
            pose = data['pose']
            trans = data['trans']
            
            # For Camera 0
            cam0_params = camera1_params
            reprojected_cam0 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam0_params)
            reprojected_cam0_list.append(reprojected_cam0.flatten())
            
            # For Camera 2
            cam2_params = camera1_params
            reprojected_cam2 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam2_params)
            reprojected_cam2_list.append(reprojected_cam2.flatten())
            
            # For Camera 3
            cam3_params = camera1_params
            reprojected_cam3 = transform_smpl_to_camera_frame(pose, trans, camera1_params, cam3_params)
            reprojected_cam3_list.append(reprojected_cam3.flatten())
            
            identifier = filename.split('/')[6]
            identifiers.append(identifier)

        return ground_SMPL_list, reprojected_cam0_list, reprojected_cam2_list, reprojected_cam3_list, identifiers

    def load_object_data(object_path, base_path):
        object_data_list = []
        reprojected_cam0_list = []
        reprojected_cam2_list = []
        reprojected_cam3_list = []
        paths = []
        identifiers = []

        for filename in glob.iglob(os.path.join(object_path, '**', '*', 'fit01', '*_fit.pkl'), recursive=True):
            paths.append(filename)

        paths = sorted(paths)
        for filename in paths:
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            
            #object_data = np.concatenate([data['angle'], data['trans']])
            #object_data = np.concatenate([data['trans']])
            object_data = np.concatenate([data['angle']])
            object_data_list.append(object_data)

            Date = filename.split('/')[6].split('_')[0]

            # Reproject to cameras 0, 2, and 3
            camera1_params = load_config(1, base_path, Date)

            # For Camera 0
            cam0_params = camera1_params
            reprojected_cam0 = transform_object_to_camera_frame(data, camera1_params, cam0_params)
            reprojected_cam0_list.append(reprojected_cam0.flatten())
            
            # For Camera 2
            cam2_params = camera1_params
            reprojected_cam2 = transform_object_to_camera_frame(data, camera1_params, cam2_params)
            reprojected_cam2_list.append(reprojected_cam2.flatten())
            
            # For Camera 3
            cam3_params = camera1_params
            reprojected_cam3 = transform_object_to_camera_frame(data, camera1_params, cam3_params)
            reprojected_cam3_list.append(reprojected_cam3.flatten())
            
            identifier = filename.split('/')[6]
            identifiers.append(identifier)

        return object_data_list, reprojected_cam0_list, reprojected_cam2_list, reprojected_cam3_list, identifiers

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
            self.manual_backward(avg_loss)
            optimizer = self.optimizers()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"loss_train": avg_loss.item()})
            return avg_loss

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
            
            return val_loss.item() # Return the average validation loss

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
            if not hasattr(self, 'validation_losses'):
                self.validation_losses = []
            self.validation_losses.append(avg_loss)
            wandb.log({"loss_val": avg_loss.item()})
            return avg_loss

        def on_validation_epoch_end(self):
            avg_val_loss = torch.mean(torch.stack(self.validation_losses))
            wandb.log({"epoch_val_loss": avg_val_loss.item()})
            self.log_scheduler_info(avg_val_loss)
            #del self.validation_losses

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
    behave_seq = '/scratch_net/biwidl307_second/lgermano/behave/sequences/Date01_Sub01_backpack_back'
    base_path = '/scratch_net/biwidl307_second/lgermano/behave'
    ground_SMPL_list, reprojected_smpl_cam0, reprojected_smpl_cam2, reprojected_smpl_cam3, ground_SMPL_identifiers = load_ground_truth_SMPL(behave_seq, base_path)
    object_data_list, reprojected_obj_cam0, reprojected_obj_cam2, reprojected_obj_cam3, object_data_identifiers = load_object_data(behave_seq, base_path)
    input_dim = ground_SMPL_list[0].shape[0]
    #output_dim = object_data_list[0].shape[0]
    output_dim = 3

    # Ensure the identifiers from both lists match
    assert ground_SMPL_identifiers == object_data_identifiers

    dataset = BehaveDataset(ground_SMPL_list, reprojected_smpl_cam0, reprojected_smpl_cam2, reprojected_smpl_cam3, object_data_list, reprojected_obj_cam0, reprojected_obj_cam2, reprojected_obj_cam3, object_data_identifiers)

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
    if model.validation_losses:
        avg_val_loss = torch.mean(torch.stack(model.validation_losses)).item()
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
    trainer.test(model, datamodule=data_module)

    # Save the model using WandB run ID
    filename = f"/scratch_net/biwidl307/lgermano/crossvit/trained_models/model_{wandb.run.name}.pt"

    # Save the model
    torch.save(model, filename)

    # Finish the current W&B run
    wandb.finish()

# After all trials, print the best set of hyperparameters
print("Best Validation Loss:", best_val_loss)
print("Best Hyperparameters:", best_params)