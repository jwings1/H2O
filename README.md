![Header Image](/reports/figures/FireflyHuman2Object.png)
*Image credits: [Firefly](https://firefly.adobe.com/generate/font-styles?prompt=Water%2C+ocean%2C+bubbles&fitType=loose&seed=36001&text=Human2Object&font=alfarn-2&bgColor=transparent&textColor=transparent&var=13865&var=17773&var=86511&dl=it-IT&ff_campaign=ffly_homepage&ff_source=firefly_seo&ff_channel=adobe_com)*

# **3D Human-Object Interaction in Video:** *A New Approach to Object Tracking via Cross-Modal Attention*

# Abstract 📚

A novel framework for 6-DoF (Six Degrees of Freedom) object tracking in RGB video is introduced, named H2O-CA (Human to Object - Cross Attention). This framework adopts a sequence-to-sequence approach: it utilizes a method for the regression of avatars to parametrically model the human body, then groups offsets in a sliding-window fashion, and employs a cross-modal attention mechanism to attend human pose to object pose.

The study commences by comparing datasets and regression methods for avatars in 5D (TRACE/ROMP/BEV/4DH) and scrutinizing various coordinate systems, including absolute, relative, and trilateration techniques, with the BEHAVE dataset being employed throughout. The significance of human pose in tracking tasks is explored by juxtaposing it with a baseline encoder model that relies solely on object pose.

Various training configurations, differentiated by their loss functions, are investigated for the tracking task. Additionally, the framework is compared with other object-tracking methodologies (DROID-SLAM/BundleTrack/KinectFusion/NICE-SLAM/SDF-2-SDF/BundleSDF). The approach is particularly effective in scenarios influenced by human actions, such as lifting or pushing, which direct object movement, and in instances of partial or full object obstructions.

Qualitative results are illustrated [here](https://jwings1.github.io/H2O-CA/). Although the fully recursive tracking approach does not achieve state-of-the-art performance, the potential of next-frame prediction and next-4 frames prediction is acknowledged. The primary application envisioned is in augmented reality (AR).

[![Read More](https://img.shields.io/badge/Read%20the%20Paper-PDF-blue.svg)](https://github.com/jwings1/H2O/blob/code-refactored/reports/3D_Human_Object_Interaction_in_Video.pdf) [![View Slides](https://img.shields.io/badge/View%20Slides-PDF-blue.svg)](https://tumde-my.sharepoint.com/:p:/g/personal/lorenzo_germano_tum_de/EUB0H4NWhSNJr6DZxvAC7DYBSbnNhMDW0_JukpY1c5SVUg?rtime=xh2BW0wu3Eg)

<p align="center">
  <img src="/reports/figures/Pipeline.png" alt="H2O-CA pipeline" width="50%">
  <br>
  <em>H2O-CA pipeline. In step 1, in a fully recursive approach, the first 8 frames of the video are equipped with an arbitrary reference frame, and successive relative offsets of the position and orientation of the object are computed. In step 2, the sliding window W in input (width 12, offset 1), and the sliding window O of offsets (width 2, offset 1) are portrayed. In step 3, a method for regression of avatars has been applied. In step 4, the regressive unit H2O-CA yields, after hot initialization (green), fully recursive predictions (light blue).</em>
</p>

# Project structure 📂

```text
User
.
├── LICENSE                              <- Open-source license 📜
├── Makefile                             <- Makefile with convenience commands like `make data` 📦
├── README.md                            <- Project description and instructions 📄
├── data
│   ├── processed                        <- The final datasets, human annotations, and data modules 📊
│   └── raw                              <- The original data dump 📥
├── environment.yml                      <- Conda environment file for ensuring reproducibility across setups 🌍 
├── h2o_ca
│   ├── H2O_CA.py                        <- Main model implementation file, see Table 4.1, column Setup, row 2 🧠
│   ├── H2O_CA_chain.py                  <- See Table 4.1, column Setup, row 3  🔄
│   ├── H2O_CA_encoder_only.py           <- Encoder-only model variant, see Table 4.1, column Setup, row 4 🧩
│   ├── H2O_CA_next_frame_loss.py        <- See Table 4.1, column Setup, row 1 🔮
│   ├── __init__.py                      <- Makes h2o_ca a Python module 🐍
│   ├── __pycache__                      <- Python cache files for faster load times ⚡
│   ├── data                             <- Scripts to generate datasets 📦
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── behave_dataset.py            <- Script with DataModule and DataLoader 🧐
│   │   ├── labels.py                    <- Script for chosing which labels to process 🏷️
│   │   ├── make_dataset.py              <- Script for creating and preprocessing datasets 🛠️
│   │   └── utils.py                     <- Utility functions for dataset preparation 🛠️
│   ├── environment.yml                  <- Environment file specific to model development 🌱
│   ├── log                              <- Logs for training and prediction processes 📝
│   ├── models                           <- Saved model checkpoints 🤖
│   │   ├── __init__.py
│   │   ├── model_encoder_only_epoch_4.pt
│   │   ├── model_radiant-leaf-3120_epoch_119.pt
│   │   ├── model_radiant-leaf-3120_epoch_99.pt
│   │   └── model_single_prediction_epoch_563.pt
│   ├── train_model.py                   <- Main script for training models 🏋️
│   ├── train_model.sh                   <- Shell script for model training automation 🚂
│   └── visualizations                   <- Scripts and resources for model predictions and visualizations 🚀
│       ├── __init__.py
│       ├── __pycache__
│       ├── metrics.py                   <- Script for calculating and reporting metrics 📏
│       ├── predict.py                   <- Script for making predictions with a trained model 🔮
│       ├── predict.sh                   <- Shell script for running predictions 🚀
│       └── videos
├── h2o_ca.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
├── pyproject.toml                       <- Project configuration file ⚙️
├── reports                              <- Reports, including figures and videos 📊
│   ├── 3D_Human_Object_Interaction_in_Video.pdf <- Report on human-object interaction analysis 📑
│   ├── figures                          <- README figures 🖼️
│   │   ├── FireflyHuman2Object.png
│   │   └── Pipeline.png
│   └── videos                           <- Directory for storing generated videos 📹
│       └── Date02_Sub02_boxsmall_hand_20240117_003809.mp4
├── requirements.txt                     <- The requirements file for reproducing the analysis environment 🐍
├── requirements_dev.txt                 <- Additional requirements for development purposes 🧪
└── trilateration
    └── robustness_of_distance.py        <- See section 3.3.4 📏
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps). 🚀

# Environment setup 🌐

## 1. To create the required environment, use the following command:

```bash
CONDA_OVERRIDE_CUDA=11.7 conda create --name pytcu11 pytorch=2.0.1 pytorch-cuda=11.7 torchvision cudatoolkit=11.7 pytorch-lightning scipy wandb matplotlib --channel pytorch --channel nvidia
```
You can also check the `environment.yml` file located at `/scratch/lgermano/H2O/environment.yml`.

Ensure that your PyTorch and CUDA versions match the compatibility matrix. Refer to [NVIDIA's Dependency Matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) for guidance on compatible versions. See additionally [here](https://computing.ee.ethz.ch/Programming/Languages/GPUCPU?highlight=%28dependency%29%7C%28matrix%29).

Missing libraries can be installed via `pip install -e .` .

# Dataset Acquisition and Setup 📊

## 1. Download the Dataset

Before using the dataset, you need to download it from the provided source. The dataset is available at [MPI Virtual Humans](https://virtualhumans.mpi-inf.mpg.de/behave/license.html). Please ensure that you have read and agreed to the license terms.

#### Download Links:
- [Scanned objects](https://virtualhumans.mpi-inf.mpg.de/behave/scanned_objects.zip)
- [Calibration files](https://virtualhumans.mpi-inf.mpg.de/behave/calibration_files.zip)
- [Train and test split](https://virtualhumans.mpi-inf.mpg.de/behave/train_test_split.zip)

1. **Template and Split File Paths**: Ensure `base_path_template` and `path_to_file` reflect your directory structure.
   ```python
   base_path_template = "/your_path_here/raw/behave/"
   path_to_file = "/your_path_here/raw/behave/"
   ```

2. **Base Path for Annotations**: Update `base_path_annotations` to where your annotations are stored.
   ```python
   base_path_annotations = "/your_path_here/raw/behave/behave-30fps-params-v1/"
   ```
  - Sequences separated by dates (in total ~140GB):
    - [Date01 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date01.zip)
    - [Date02 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date02.zip)
    - [Date03 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date03.zip)
    - [Date04 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date04.zip)
    - [Date05 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date05.zip)
    - [Date06 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date06.zip)
    - [Date07 sequences](https://virtualhumans.mpi-inf.mpg.de/behave/Date07.zip)

    #### Unzipping Sequence Files

    After downloading all the sequences, you can extract them using the following command:

    ```bash
    unzip "Date*.zip" -d sequences
    ```

3. **Base Path for TRACE Results (or the method of choice)**: Modify `base_path_trace` if your TRACE results are stored in a different location.
   ```python
   base_path_trace = "/your_path_here/data/processed/TRACE_results"
   ```
4. **Dataset File Path**: Change the `data_file_path` to retrieve a generated dataset.
   ```python
   data_file_path = "/your_path_here/data/processed/datasets/your_dataset_here.pkl"
   ```

# Training Setup 🏋️‍♂️

## 1. Cluster Job Submission Guide

#### SLURM Script Template

Below are parts of SLURM script train_model.sh. Ensure you replace the placeholders with the actual paths relevant to your setup.

```bash
#!/bin/bash

#SBATCH --job-name="train model"
#SBATCH --error=/your_path_here/H2O/h2o_ca/log/error/%j.err
#SBATCH --output=/your_path_here/H2O/h2o_ca/log/out/%j.out

# Set up the Conda environment
source /your_conda_path_here/etc/profile.d/conda.sh
conda activate evaluation

# Set necessary environment variables
export PYTHONPATH=/your_path_here/smplpytorch/smplpytorch:$PYTHONPATH
export CONDA_OVERRIDE_CUDA=11.8
export WANDB_DIR=/your_path_here/H2O/h2o_ca/log/cache

# Execute the Python training script
python /your_path_here/H2O/H2O_ca/train_model.py "$@"
```

#### Adjusting Paths

- **SBATCH Directives**: Adjust the paths in `--error` and `--output` to point to your log directories.
- **Conda Activation**: Replace `/your_conda_path_here/etc/profile.d/conda.sh` with the path where your Conda is initialized.
- **Environment Variables**:
  - `PYTHONPATH`: Update with the path to your Python modules or packages if necessary.
  - `WANDB_DIR`: Set this to the directory where you want Weights & Biases to store its logs.
- **Python Script Execution**: Change the path in the `python` command to where your training script is located.

## 2. Command-Line Interface Options
The following CLI options are available for configuring the training process:

### Model and Data Configuration Options

- `--first_option`: Specify input to encoder in the orientation branch. For example, choices may include `SMPL_pose`, `pose_trace`, `unrolled_pose`, `unrolled_pose_trace`, `enc_unrolled_pose`, `enc_unrolled_pose_trace`.
  
- `--second_option`: Specify input to encoder in the position branch. For example, choices may include `SMPL_joints`, `distances`, `joints_trace`, `norm_joints`, `norm_joints_trace`, `enc_norm_joints`, `enc_norm_joints_trace`.
  
- `--third_option`: Choose e.g. between `OBJ_pose` and `enc_obj_pose` for input to the decoder in the orientation branch.
  
- `--fourth_option`: Defines input to  the decoder in the position branch e.g. with choices `OBJ_trans`, `norm_obj_trans`, `enc_norm_obj_trans`.

- `--scene`: Include scene information in the options. Default is `Date01_Sub01_backpack_back`.

See https://github.com/jwings1/3DObjTracking/tree/master for a comparison of methods of regressing avatars.

### Training Configuration Options

- `--learning_rate`: Set the learning rate(s) for training. Accepts multiple values for experiments. Default is `0.0001`.
  
- `--epochs`: Number of epochs for training. Can specify multiple values. Default is `20`.
  
- `--batch_size`: Batch size for training. Accepts multiple values. Default is `16`.
  
- `--dropout_rate`: Dropout rate for the model. Accepts multiple values. Default is `0.05`.
  
- `--lambda_1`: Weight for the pose_loss. Default is `1`.
  
- `--lambda_2`: Weight for the trans_loss. Default is `1`.
  
- `--optimizer`: Choose the optimizer for training. Options are `AdamW`, `Adagrad`, `Adadelta`, `LBFGS`, `Adam`, `RMSprop`. Default is `AdamW`.

### Miscellaneous Options

- `--name`: Set a name for the training run, which will default to a timestamp.
  
- `--frames_subclip`: Number of frames per subclip. Default is `12`.
  
- `--masked_frames`: Number of masked frames. Default is `4`.
  
- `--L`: Number of interpolation frames L. Default is `1`.

- `--create_new_dataset`: Enable this option to create a new dataset for training.
  
- `--load_existing_dataset`: Enable this option to load an existing dataset for training.
  
- `--save_data_module`: Specify whether to save the data module after processing.
  
- `--load_data_module`: Specify whether to load the data module. Default is enabled.

- `--cam_ids`: Camera IDs used for training. Accepts multiple values. Default is `1`.

## 3. Running the job

You can explore certain hyperparameters through a grid search by setting their ranges as flags, as shown in the example:

```bash
sbatch train_model.sh --first_option='pose' --second_option='joints' --third_option='obj_pose' --fourth_option='obj_trans' --name='block_cam2' --L=[1,4]
```

## 4. Monitoring Your Job

After adjusting the paths in the SLURM script, monitor your job's progress through the SLURM utilities (`squeue`, `sacct`, etc.) and the log files specified in the SBATCH directives.

The execution should call `/scratch/lgermano/H2O/h2o_ca/data/make_dataset.py` to create and store data in `/scratch/lgermano/H2O/data/raw` or retrieve it, then save it into `/scratch/lgermano/H2O/data/processed`. The entire BEHAVE dataset takes up 4 GB. Choose the labels to train and pick the architecture you want to train in `train_model`. Optionally, you can initialize the model with old checkpoints at `/scratch/lgermano/H2O/h2o_ca/models`.

### Dataset Usage Example

To access and utilize the dataset for research or application development, you can follow this Python code snippet:

```python
# Assuming 'data' is your dataset loaded from the pickle file
num_camera_views = len(data)
print(f"Number of camera views in the dataset: {num_camera_views}")

# Accessing data from the first camera view
first_camera_view_data = data[0]
num_frames_first_view = len(first_camera_view_data)
print(f"Number of frames in the first camera view: {num_frames_first_view}")

# Accessing the first frame in the first camera view
first_frame_data = first_camera_view_data[0]
frame_keys = first_frame_data.keys()
print(f"Data keys available in a frame: {frame_keys}")

```
# Makefile utilities 🇲

```bash
make create_environment
conda activate h2o_ca
make requirements  # install everything in the requirements.txt file
make dev_requirements
make clean  # clean __pycache__ files
make data  # runs the make_dataset.py file
```

# Bibtex 📝

```bibtex
@misc{Germano_2024,
  author       = {Germano},
  title        = {3D Human-Object Interaction in Video: A New Approach to Object Tracking via Cross-Modal Attention},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/jwings1/H2O/tree/code-refactored}},
  commit       = {GitHubCommitHash},
  note         = {Accessed: Access Date}
}

```

# Contact Information 📬

For any inquiries, issues, or contributions, please contact:

**Lorenzo Germano**  
- 📧 Email: [lorenzogermano1@outlook.it](mailto:lorenzogermano1@outlook.it)
- 🔗 LinkedIn: [lorenzogermano](https://www.linkedin.com/in/lorenzogermano/)
