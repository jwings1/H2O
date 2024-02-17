import os
import gc
import argparse
import itertools
from datetime import datetime
import pdb
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import pytorch_lightning as pl
import wandb


class CustomCosineLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CustomCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class CombinedTrans(pl.LightningModule):
    def __init__(self, frames_subclip, masked_frames):
        super(CombinedTrans, self).__init__()

        self.save_hyperparameters()
        self.validation_losses = []
        self.frames_subclip = frames_subclip
        self.masked_frames = masked_frames
        self.automatic_optimization = False
        self.best_avg_loss_val = float("inf")

        self.num_heads = 4
        self.d_model = 128
        self.mlp_output_pose = MLP(self.d_model, 3)
        self.mlp_output_trans = MLP(self.d_model, 3)
        self.mlp_smpl_pose = MLP(72, self.d_model)
        self.mlp_smpl_joints = MLP(72, self.d_model)
        self.mlp_obj_pose = MLP(3, self.d_model)
        self.mlp_obj_trans = MLP(3, self.d_model)

        self.transformer_model_trans = nn.Transformer(
            d_model=self.d_model,
            nhead=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.05,
            activation="gelu",
        )

        self.transformer_model_trans = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=0.05,
            activation="gelu",
        )

        self.transformer_model_pose = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=0.05,
            activation="gelu",
        )

    def forward(self, smpl_pose, smpl_joints, obj_pose, obj_trans):
        # smpl_pose, smpl_joints, obj_pose, obj_trans = cam_data[-2][:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print("SMPL Pose:", smpl_pose.shape)
        # print("SMPL Joints:", smpl_joints.shape)
        # print("Object Pose:", obj_pose.shape)
        # print("Object Trans:", obj_trans.shape)

        def positional_encoding(dim, sentence_length):
            """
            Creates a positional encoding as used in Transformer models.
            :param dim: Embedding size
            :param sentence_length: The length of the input sequence
            :return: A tensor of shape [sentence_length, dim] with the positional encoding.
            """
            # Initialize a matrix of zeros
            encoding = np.zeros((sentence_length, dim))

            # Calculate the positional encodings
            for pos in range(sentence_length):
                for i in range(0, dim, 2):
                    encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / dim)))
                    encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / dim)))

            return torch.tensor(encoding, dtype=torch.float32)

        # Create positional encoding
        pos_encoding = positional_encoding(self.d_model, self.frames_subclip).unsqueeze(0).to(device)

        # Embedding inputs
        # embedded_smpl_pose = self.mlp_smpl_pose(smpl_pose) + pos_encoding
        embedded_obj_pose = self.mlp_obj_pose(obj_pose) + pos_encoding
        # embedded_smpl_joints = self.mlp_smpl_joints(smpl_joints) + pos_encoding
        embedded_obj_trans = self.mlp_obj_trans(obj_trans) + pos_encoding

        # Masking
        embedded_obj_pose[:, -self.masked_frames :, :] = 0
        embedded_obj_trans[:, -self.masked_frames :, :] = 0

        predicted_obj_pose_emb = self.transformer_model_pose(embedded_obj_pose.permute(1, 0, 2))
        predicted_obj_trans_emb = self.transformer_model_trans(embedded_obj_trans.permute(1, 0, 2))

        predicted_obj_pose = self.mlp_output_pose(predicted_obj_pose_emb.permute(1, 0, 2))
        predicted_obj_trans = self.mlp_output_trans(predicted_obj_trans_emb.permute(1, 0, 2))

        return predicted_obj_pose, predicted_obj_trans

    def training_step(self, cam_data):
        def axis_angle_to_rotation_matrix(axis_angle):
            # Ensure axis_angle is a batched input
            batch_size, masked_frames, _ = axis_angle.shape

            # Normalize the axis part of the axis-angle vector
            axis = F.normalize(axis_angle, dim=-1)
            angle = torch.norm(axis_angle, dim=-1, keepdim=True)

            # Get the skew-symmetric cross-product matrix of the axis
            skew = torch.zeros((batch_size, masked_frames, 3, 3), device=axis_angle.device)
            skew[:, :, 0, 1] = -axis[:, :, 2]
            skew[:, :, 1, 0] = axis[:, :, 2]
            skew[:, :, 0, 2] = axis[:, :, 1]
            skew[:, :, 2, 0] = -axis[:, :, 1]
            skew[:, :, 1, 2] = -axis[:, :, 0]
            skew[:, :, 2, 1] = axis[:, :, 0]

            # Rodrigues' rotation formula
            I = (
                torch.eye(3, device=axis_angle.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, masked_frames, -1, -1)
            )
            sin_angle = torch.sin(angle).unsqueeze(-1)  # Shape (batch_size, masked_frames, 1, 1)
            cos_angle = (1 - torch.cos(angle)).unsqueeze(-1)  # Shape (batch_size, masked_frames, 1, 1)
            R = I + sin_angle * skew + cos_angle * torch.matmul(skew, skew)

            return R

        def rotation_matrix_to_6d(R):
            # Extract the first two columns of the rotation matrix to form the 6D representation
            return R[..., :2].reshape(-1, 6)

        def rotation_6d_loss(rot1, rot2):
            # Convert axis-angle to rotation matrix
            R1 = axis_angle_to_rotation_matrix(rot1)
            R2 = axis_angle_to_rotation_matrix(rot2)

            # Convert rotation matrix to 6D representation
            rot_6d_1 = rotation_matrix_to_6d(R1)
            rot_6d_2 = rotation_matrix_to_6d(R2)

            # Compute the loss as mean squared error between the two 6D representations
            loss = F.mse_loss(rot_6d_1, rot_6d_2)
            return loss

        # Normalization function
        def normalize_data(data, mean, std):
            return (data - mean) / std

        def unnormalize_data(data, mean, std):
            return (data * std) + mean

        def train_one_instance(smpl_pose, smpl_joints, obj_pose, obj_trans, GT_obj_pose, GT_obj_trans):
            optimizer.zero_grad()

            # Normalize the batched data during traing. Trans only for now.
            smpl_joints_mean = torch.ones(smpl_joints.size()) * 1e-2
            smpl_joints_std = torch.ones(smpl_joints.size()) * 1e-0
            obj_trans_mean = torch.ones(obj_trans.size()) * 1e-2
            obj_trans_std = torch.ones(obj_trans.size()) * 1e-0

            norm_smpl_joints = normalize_data(
                smpl_joints.to(device), smpl_joints_mean.to(device), smpl_joints_std.to(device)
            )
            norm_obj_trans = normalize_data(obj_trans.to(device), obj_trans_mean.to(device), obj_trans_std.to(device))

            masked_obj_pose = obj_pose.clone()
            masked_obj_trans = norm_obj_trans.clone()

            smpl_pose = smpl_pose.to(device)
            smpl_joints = norm_smpl_joints.to(device)
            masked_obj_pose = masked_obj_pose.to(device)
            masked_obj_trans = masked_obj_trans.to(device)
            obj_pose = obj_pose.to(device)
            obj_trans = norm_obj_trans.to(device)

            predicted_obj_pose, predicted_obj_trans = self.forward(
                smpl_pose, smpl_joints, masked_obj_pose, masked_obj_trans
            )

            GT_obj_pose = GT_obj_pose.unsqueeze(0)
            GT_obj_trans = GT_obj_trans.unsqueeze(0)

            pose_loss = rotation_6d_loss(
                predicted_obj_pose[:, -self.masked_frames :, :], GT_obj_pose[:, -self.masked_frames :, :]
            )
            trans_loss = F.mse_loss(
                predicted_obj_trans[:, -self.masked_frames :, :], GT_obj_trans[:, -self.masked_frames :, :]
            )

            total_loss = pose_loss + trans_loss
            mean_total_loss = torch.mean(total_loss)

            # Logging the losses
            self.log("train_pose_loss", pose_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("train_trans_loss", trans_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(
                "mean_train_total_loss",
                mean_total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.manual_backward(mean_total_loss)
            optimizer.step()

            # Unnormalize
            predicted_obj_trans = unnormalize_data(
                predicted_obj_trans.to(device), obj_trans_mean.to(device), obj_trans_std.to(device)
            )

            return predicted_obj_pose.detach(), predicted_obj_trans.detach()

        # Backward pass and optimization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = self.optimizers()

        smpl_pose, smpl_joints, GT_obj_pose, GT_obj_trans = cam_data[-2][:]
        smpl_joints = smpl_joints.reshape(-1, self.frames_subclip, 72)

        for i in range(smpl_pose.shape[0]):
            if i == 0:
                # The initial window is GT. Dimension reduced.
                obj_pose = GT_obj_pose[0].clone()
                obj_trans = GT_obj_trans[0].clone()

            # Inputs should be batched
            predicted_obj_pose, predicted_obj_trans = train_one_instance(
                smpl_pose[i], smpl_joints[i], obj_pose, obj_trans, GT_obj_pose[i], GT_obj_trans[i]
            )

            # Update obj_pose, obj_trans. Roll along the window dimension. dim = 3

            obj_pose = torch.roll(obj_pose, -1, 0)
            obj_pose[-self.masked_frames - 1, :] = predicted_obj_pose[:, -self.masked_frames, :]

            obj_trans = torch.roll(obj_trans, -1, 0)
            obj_trans[-self.masked_frames - 1, :] = predicted_obj_trans[:, -self.masked_frames, :]

        return None

    def validation_step(self, cam_data, batch_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smpl_pose, smpl_joints, obj_pose, obj_trans = cam_data[-2][:]
        smpl_joints = smpl_joints.reshape(-1, self.frames_subclip, 72)

        # Normalization function
        def normalize_data(data, mean, std):
            return (data - mean) / std

        # Normalize the batched data during traing. Trans only for now.
        smpl_joints_mean = torch.ones(smpl_joints.size()) * 1e-2
        smpl_joints_std = torch.ones(smpl_joints.size()) * 1e-0
        obj_trans_mean = torch.ones(obj_trans.size()) * 1e-2
        obj_trans_std = torch.ones(obj_trans.size()) * 1e-0

        norm_smpl_joints = normalize_data(
            smpl_joints.to(device), smpl_joints_mean.to(device), smpl_joints_std.to(device)
        )
        norm_obj_trans = normalize_data(obj_trans.to(device), obj_trans_mean.to(device), obj_trans_std.to(device))

        masked_obj_pose = obj_pose.clone()
        masked_obj_trans = norm_obj_trans.clone()

        # Move each tensor to the specified device
        smpl_pose = smpl_pose.to(device)
        smpl_joints = norm_smpl_joints.to(device)
        masked_obj_pose = masked_obj_pose.to(device)
        masked_obj_trans = masked_obj_trans.to(device)
        obj_pose = obj_pose.to(device)
        obj_trans = norm_obj_trans.to(device)

        # # Assuming predictions contain the masked_obj_pose and masked_obj_trans
        # predicted_obj_pose, predicted_obj_trans = self.forward(smpl_pose, smpl_joints, masked_obj_pose, masked_obj_trans)

        # Assuming smpl_pose, smpl_joints, masked_obj_pose, and masked_obj_trans are batched tensors
        batch_size = smpl_pose.size(0)

        # Initialize lists to store the results for each instance in the batch
        predicted_obj_poses = []
        predicted_obj_transes = []

        # Iterate over each instance in the batch
        for i in range(batch_size):
            # Extract the i-th instance from each tensor
            instance_smpl_pose = smpl_pose[i].unsqueeze(0)  # Add batch dimension
            instance_smpl_joints = smpl_joints[i].unsqueeze(0)
            instance_masked_obj_pose = masked_obj_pose[i].unsqueeze(0)
            instance_masked_obj_trans = masked_obj_trans[i].unsqueeze(0)

            # Forward pass for the single instance
            instance_predicted_obj_pose, instance_predicted_obj_trans = self.forward(
                instance_smpl_pose, instance_smpl_joints, instance_masked_obj_pose, instance_masked_obj_trans
            )

            # Store the results
            predicted_obj_poses.append(instance_predicted_obj_pose)
            predicted_obj_transes.append(instance_predicted_obj_trans)

        # Concatenate the results to form a batch again
        predicted_obj_pose = torch.cat(predicted_obj_poses, dim=0)
        predicted_obj_trans = torch.cat(predicted_obj_transes, dim=0)

        # Undo normalization

        def unnormalize_data(data, mean, std):
            return (data * std) + mean

        predicted_obj_trans = unnormalize_data(
            predicted_obj_trans.to(device), obj_trans_mean.to(device), obj_trans_std.to(device)
        )

        # Compute L2 loss (MSE) for both pose and translation
        # smpl_pose_loss = F.mse_loss(predicted_smpl_pose, smpl_pose)
        # smpl_joints_loss = F.mse_loss(predicted_smpl_joints, smpl_joints.reshape(wandb.config.batch_size, -1, 72))
        pose_loss = F.mse_loss(predicted_obj_pose[:, -self.masked_frames, :], obj_pose[:, -self.masked_frames, :])
        trans_loss = F.mse_loss(predicted_obj_trans[:, -self.masked_frames, :], obj_trans[:, -self.masked_frames, :])

        # Combine the losses
        total_loss = pose_loss + trans_loss
        # total_loss = trans_loss

        # self.validation_losses.append(total_loss)
        self.validation_losses.append(total_loss)

        # Log the losses. The logging method might differ slightly based on your framework
        self.log("val_pose_loss", pose_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_smpl_pose_loss', smpl_pose_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_smpl_joints_loss', smpl_joints_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_trans_loss", trans_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # # Logging the losses to wandb
        # wandb.log({
        #     'val_total_loss': total_loss.item(),
        #     'val_pose_loss': pose_loss.item(),
        #     #'val_smpl_pose_loss': smpl_pose_loss.item(),
        #     #'val_smpl_joints_loss': smpl_joints_loss.item(),
        #     'val_trans_loss': trans_loss.item(),
        # })

        return {"val_loss": total_loss}

    def on_validation_epoch_end(self):
        avg_val_loss = torch.mean(torch.tensor(self.validation_losses))

        # wandb.log({
        #     'Val Trans+Angle Epoch-Averaged Batch-Averaged Average 4Cameras': avg_val_loss.item()
        #     })l
        self.log("avg_val_loss", avg_val_loss, prog_bar=True, logger=True)
        wandb.log({"Learning Rate": self.optimizer.param_groups[0]["lr"]})
        if avg_val_loss < self.best_avg_loss_val:
            self.best_avg_loss_val = avg_val_loss
            print(f"Best number of epochs:{self.current_epoch}")

            # Save the model
            model_save_path = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/model_{wandb.run.name}_epoch_{self.current_epoch}.pt"
            torch.save(self.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        self.validation_losses = []  # reset for the next epoch
        self.lr_scheduler.step(avg_val_loss)  # Update

    def test_step(self, batch, batch_idx):
        # smpl_pose, smpl_joints, masked_obj_pose, masked_obj_trans, obj_pose, obj_trans = batch

        # # Forward pass
        # predicted_obj_pose, predicted_obj_trans = self.forward(batch)

        # # Combine the losses
        # total_loss = pose_loss + trans_loss

        # # Logging the losses
        # self.log('test_pose_loss', pose_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_trans_loss', trans_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return 0

    def configure_optimizers(self):
        if wandb.config.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=1e-4
            )
        elif wandb.config.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(self.parameters(), lr=wandb.config.learning_rate, weight_decay=1e-4)
        elif wandb.config.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(),
                lr=wandb.config.learning_rate,
                alpha=0.99,
                eps=1e-08,
                weight_decay=1e-4,
                momentum=0.9,
            )
        elif wandb.config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4
            )
        elif wandb.config.optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(
                self.parameters(), lr=wandb.config.learning_rate, rho=0.9, eps=1e-06, weight_decay=1e-4
            )
        elif wandb.config.optimizer == "LBFGS":
            optimizer = torch.optim.LBFGS(
                self.parameters(), lr=wandb.config.learning_rate, max_iter=20, line_search_fn="strong_wolfe"
            )
        else:  # default to Adam if no match
            optimizer = torch.optim.Adam(
                self.parameters(), lr=wandb.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4
            )

        # scheduler = {
        #     'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True, factor=0.01, threshold=0.75, threshold_mode='rel'),
        #     'monitor': 'avg_val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1
        # }

        # scheduler = {
        # 'scheduler': CustomCyclicLR(optimizer, base_lr=1e-7, max_lr=5e-3, step_size=1, mode='exp_range'),
        # 'interval': 'epoch',  # step-based updates i.e. batch
        # #'monitor' : 'avg_val_loss',
        # 'name': 'custom_clr'
        # }

        scheduler = {
            "scheduler": CustomCosineLR(optimizer, T_max=100, eta_min=1e-7),
            "interval": "step",  # epoch-based updates
            "monitor": "avg_val_loss",
            "name": "custom_cosine_lr",
        }

        self.optimizer = optimizer  # store optimizer as class variable for logging learning rate
        self.lr_scheduler = scheduler[
            "scheduler"
        ]  # store scheduler as class variable for updating in on_validation_epoch_end
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
