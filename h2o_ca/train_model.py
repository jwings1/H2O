import os
import gc
import argparse
import itertools
from datetime import datetime
import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from data.labels import labels
from data.behave_dataset import BehaveDatasetOffset, BehaveDataModule
from data.utils import *

# Load architecture
from H2O_CA import CombinedTrans

# Function to create a timestamp
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to create a command-line argument parser
def create_parser():
    parser = argparse.ArgumentParser(description="Training script for H20 model.")
    parser.add_argument(
        "--first_option",
        choices=[
            "SMPL_pose",
            "pose_trace",
            "unrolled_pose",
            "unrolled_pose_trace",
            "enc_unrolled_pose",
            "enc_unrolled_pose_trace",
        ],
        help="Specify the first option.",
    )
    parser.add_argument(
        "--second_option",
        choices=[
            "SMPL_joints",
            "distances",
            "joints_trace",
            "norm_joints",
            "norm_joints_trace",
            "enc_norm_joints",
            "enc_norm_joints_trace",
        ],
        help="Specify the second option.",
    )
    parser.add_argument("--third_option", choices=["OBJ_pose", "enc_obj_pose"], help="Specify the third option.")
    parser.add_argument(
        "--fourth_option",
        choices=["OBJ_trans", "norm_obj_trans", "enc_norm_obj_trans"],
        help="Specify the fourth option.",
    )
    parser.add_argument("--scene", default=["scene"], help="Include scene in the options.")
    parser.add_argument("--learning_rate", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--epochs", nargs="+", type=int, default=[12])
    parser.add_argument("--batch_size", nargs="+", type=int, default=[16])
    parser.add_argument("--dropout_rate", nargs="+", type=float, default=[0.05])
    parser.add_argument("--lambda_1", nargs="+", type=float, default=[1], help="Weight for pose_loss.")
    parser.add_argument("--lambda_2", nargs="+", type=float, default=[1], help="Weight for trans_loss.")
    parser.add_argument("--L", nargs="+", type=int, default=[1], choices=[4])
    parser.add_argument(
        "--optimizer",
        nargs="+",
        default=["AdamW"],
        choices=["AdamW", "Adagrad", "Adadelta", "LBFGS", "Adam", "RMSprop"],
    )
    parser.add_argument("--layer_sizes_1", nargs="+", type=int, default=[[256, 256, 256]])
    parser.add_argument("--layer_sizes_3", nargs="+", type=int, default=[[64, 128, 256, 128, 64]])
    parser.add_argument("--name", default=timestamp())
    parser.add_argument("--frames_subclip", type=int, default=12, help="Number of frames per subclip.")
    parser.add_argument("--masked_frames", type=int, default=4, help="Number of masked frames.")
    parser.add_argument("--L", type=int, default=1, help="Number of interpoaltion frames L.")
    parser.add_argument("--create_new_dataset", action='store_true', help="Whether to create a new dataset.")
    parser.add_argument("--load_existing_dataset", action='store_true', help="Whether to load an existing dataset.")
    parser.add_argument("--save_data_module", action='store_true', help="Whether to save the data module.")
    parser.add_argument("--load_data_module", action='store_true', help="Whether to load the data module.")
    parser.add_argument("--cam_ids", nargs="+", type=int, default=[1], help="Camera IDs used for training.")

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Set the WANDB_CACHE_DIR environment variable
    os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"

    # Assign parsed values to corresponding variables
    first_option = args.first_option
    second_option = args.second_option
    third_option = args.third_option
    fourth_option = args.fourth_option
    scene = args.scene
    learning_rate_range = args.learning_rate
    epochs_range = args.epochs
    batch_size_range = args.batch_size
    dropout_rate_range = args.dropout_rate
    lambda_1_range = args.lambda_1
    lambda_2_range = args.lambda_2
    L_range = args.L
    optimizer_list = args.optimizer
    layer_sizes_range_1 = args.layer_sizes_1
    layer_sizes_range_3 = args.layer_sizes_3
    name = args.name

    for (
        lr,
        bs,
        dr,
        layers_1,
        layers_3,
        lambda_1,
        lambda_2,
        l,
        epochs,
        optimizer_name,
    ) in itertools.product(
        learning_rate_range,
        batch_size_range,
        dropout_rate_range,
        layer_sizes_range_1,
        layer_sizes_range_3,
        lambda_1_range,
        lambda_2_range,
        L_range,
        epochs_range,
        optimizer_list,
    ):
        # Set hyperparameters for training
        LEARNING_RATE = lr
        BATCH_SIZE = bs
        DROPOUT_RATE = dr
        LAYER_SIZES_1 = layers_1
        LAYER_SIZES_3 = layers_3
        INITIAL_OBJ_PRED = torch.rand((BATCH_SIZE, 24))
        LAMBDA_1 = lambda_1
        LAMBDA_2 = lambda_2
        EPOCHS = epochs
        L = l
        OPTIMIZER = optimizer_name

        # Initialize WandB run
        wandb.init(
            project="MLP",
            config={
                "learning_rate": LEARNING_RATE,
                "architecture": "H2O_CA",
                "dataset": "BEHAVE",
                "batch_size": BATCH_SIZE,
                "dropout_rate": DROPOUT_RATE,
                "layer_sizes_1": LAYER_SIZES_1,
                "layer_sizes_3": LAYER_SIZES_3,
                "lambda_1": LAMBDA_1,
                "lambda_2": LAMBDA_2,
                "L": L,
                "epochs": EPOCHS,
                "optimizer": OPTIMIZER,
                "frames_subclip": args.frames_subclip,
                "masked_frames": args.masked_frames,
                "create_new_dataset": args.create_new_dataset,
                "load_existing_dataset": args.load_existing_dataset,
                "save_data_module": args.save_data_module,
                "load_data_module": args.load_data_module,
                "cam_ids": args.cam_ids,
                "first_option": args.first_option,
                "second_option": args.second_option,
                "third_option": args.third_option,
                "fourth_option": args.fourth_option
            },
            mode="offline"
        )

        # Automatically use CUDA if available, otherwise fall back to CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the model from a checkpoint if needed
        # checkpoint_path = "your_checkpoint_path.pt"
        # checkpoint = torch.load(checkpoint_path, map_location=device)
        # model_combined.load_state_dict(checkpoint)
        
        best_overall_avg_loss_val = float("inf")
        best_params = None
        frames_subclip = 12
        masked_frames = 4

        # Instantiate with custom parameters
        #data_loader = CustomDataModule(frames_subclip=10, masked_frames=3, l=wandb.config.L)
        #data_module = data_loader.data_module
        
        # Get the data module for training or validation
        #data_module = custom_data_module.data_module()

        # Initialize your model and move it to the appropriate device
        model_combined = CombinedTrans(frames_subclip=frames_subclip, masked_frames=masked_frames)
        model_combined.to(device)

        wandb_logger = WandbLogger()
        wandb_logger.watch(model_combined, log=None, log_freq=10)

        # Initialize Trainer
        print("\nTraining\n", flush=True)
        trainer = pl.Trainer(
            max_epochs=wandb.config.epochs,
            logger=wandb_logger,
            num_sanity_val_steps=0,
        )

        # Load data 
        from data.make_dataset import data_module

        trainer.fit(model_combined, data_module)

        # Get the current timestamp and format it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Incorporate the timestamp into the filename
        filename = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/{wandb.run.name}_{timestamp}.pt"

        # Save the model
        torch.save(model_combined.state_dict(), filename)

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
                "lambda_1": LAMBDA_1,
                "lambda_2": LAMBDA_2,
                "L": L,
                "epochs": EPOCHS
            }

            # Optionally, to test the model:
            # trainer.test(combined_model, datamodule=data_module)

            # Save the model using WandB run ID
            # filename = f"/scratch_net/biwidl307_second/lgermano/H2O/trained_models/{wandb.run.name}.pt"
            filename = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/{wandb.run.name}.pt"

            # Save the model
            torch.save(model_combined, filename)

        # Finish the current W&B run
        wandb.finish()

    # After all trials, print the best set of hyperparameters
    print("Best Validation Loss:", best_overall_avg_loss_val)
    print("Best Hyperparameters:", best_params)
