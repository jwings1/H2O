![Header Image](/scratch/lgermano/H2O/reports/figures/Human2Object-2-6-2024.png)
*Image credits: [Font generator](https://www.textstudio.com/)*

# 3D Human-Object Interaction in Video: A New Approach to Object Tracking via Cross-Modal Attention 🤖📷

A novel framework for 6-DoF (Six Degrees of Freedom) object tracking in RGB video is introduced, named H2O-CA (Human to Object -- Cross Attention). This framework adopts a sequence-to-sequence approach: it utilizes a method for the regression of avatars to parametrically model the human body, then groups offsets in a sliding-window fashion, and employs a cross-modal attention mechanism to attend human pose to object pose.

The study commences by comparing datasets and regression methods for avatars in 5D (TRACE/ROMP/BEV/4DH) and scrutinizing various coordinate systems, including absolute, relative, and trilateration techniques, with the BEHAVE dataset being employed throughout. The significance of human pose in tracking tasks is explored by juxtaposing it with a baseline encoder model that relies solely on object pose.

Various training configurations, differentiated by their loss functions, are investigated for the tracking task. Additionally, the framework is compared with other object-tracking methodologies (DROID-SLAM/BundleTrack/KinectFusion/NICE-SLAM/SDF-2-SDF/BundleSDF). The approach is particularly effective in scenarios influenced by human actions, such as lifting or pushing, which direct object movement, and in instances of partial or full object obstructions.

Qualitative results are illustrated [here](https://jwings1.github.io/H2O-CA/). Although the fully recursive tracking approach does not achieve state-of-the-art performance, the potential of next-frame prediction and next-4 frames prediction is acknowledged. The primary application envisioned is in augmented reality (AR).

[![Read the Paper](https://img.shields.io/badge/Read%20the%20Paper-PDF-blue.svg)](URL_to_PDF)


```markdown

## Project structure 📂

The directory structure of the project looks like this:

```txt
├── Makefile             <- Makefile with convenience commands like `make data` or `make train` 📦
├── README.md            <- The top-level README for developers using this project. 📚
├── data
│   ├── processed        <- The final, canonical data sets for modeling. 📊
│   └── raw              <- The original, immutable data dump. 📥
│
├── docs                 <- Documentation folder 📃
│   │
│   ├── index.md         <- Homepage for your documentation 🏠
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs ⚙️
│   │
│   └── source/          <- Source directory for documentation files 📝
│
├── models               <- Trained and serialized models, model predictions, or model summaries 🤖
│
├── notebooks            <- Jupyter notebooks 📓
│
├── pyproject.toml       <- Project configuration file ⚙️
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc. 📊
│   └── figures          <- Generated graphics and figures to be used in reporting 📈
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment 🐍
│
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment 🧪
│
├── tests                <- Test files 🧪
│
├── h2o_ca  <- Source code for use in this project. 📁
│   │
│   ├── __init__.py      <- Makes folder a Python module 🐍
│   │
│   ├── data             <- Scripts to download or generate data 📦
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script 🤖
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations 📊
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model 🚂
│   └── predict_model.py <- script for predicting from a model 🚀
│
└── LICENSE              <- Open-source license if one is chosen 📜
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps). 🚀

## Build Environment

To create the required environment, use the following command:

```bash
CONDA_OVERRIDE_CUDA=11.7 conda create --name pytcu11 pytorch=2.0.1 pytorch-cuda=11.7 torchvision cudatoolkit=11.7 pytorch-lightning scipy wandb matplotlib --channel pytorch --channel nvidia
```
You can also check the `environment.yml` file located at `/scratch/lgermano/H2O/environment.yml`.

Ensure that your PyTorch and CUDA versions match the compatibility matrix. Refer to [NVIDIA's Dependency Matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) for guidance on compatible versions.

Missing libraries can be installed via `pip install`.


## Dataset Acquisition and Setup 📦

### 1. Downloading the Dataset 📥

Before using the dataset, you need to download it from the provided source. The dataset is available at [MPI Virtual Humans](https://virtualhumans.mpi-inf.mpg.de/behave/license.html). Please ensure that you have read and agreed to the license terms. 📝

#### Download Links:
- [Scanned objects](https://virtualhumans.mpi-inf.mpg.de/behave/scanned_objects.zip)
- [Calibration files](https://virtualhumans.mpi-inf.mpg.de/behave/calibration_files.zip)
- [Train and test split](https://virtualhumans.mpi-inf.mpg.de/behave/train_test_split.zip)
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

#### 2. Set up paths

For dataset creation, paths to various resources in make_dataset.py are required. Here are the defaults used in the project:

- **GT Annotations**: `/scratch_net/biwidl307_second/lgermano/behave/behave-30fps-params-v1`
- **TRACE Results**: `/srv/beegfs02/scratch/3dhumanobjint/data/TRACE_results`
- **Object Template**: `/scratch_net/biwidl307_second/lgermano/behave`
- **BEHAVE Split File**: `/scratch_net/biwidl307_second/lgermano/behave/split.json`

#### Adjusting Paths

Please adjust the paths according to your project's storage and organization:

1. **Dataset File Path**: Change the `data_file_path` to retrieve a generated dataset
   ```python
   data_file_path = "/your_path_here/datasets/your_dataset_here.pkl"
   ```

2. **Base Path for Annotations**: Update `base_path_annotations` to where your annotations are stored.
   ```python
   base_path_annotations = "/your_path_here/behave/behave-30fps-params-v1"
   ```

3. **Base Path for TRACE Results (or the method of choice)**: Modify `base_path_trace` if your TRACE results are stored in a different location.
   ```python
   base_path_trace = "/your_path_here/data/TRACE_results"
   ```

4. **Template and Split File Paths**: Ensure `base_path_template` and `path_to_file` reflect your directory structure.
   ```python
   base_path_template = "/your_path_here/behave"
   path_to_file = "/your_path_here/behave/split.json"
   ```

## Training 

### Running the job

You can explore certain hyperparameters through a grid search by setting their ranges as flags, as shown in the example:

```bash
sbatch H2O_train6_object.sh --first_option='pose' --second_option='joints' --third_option='obj_pose' --fourth_option='obj_trans' --name='block_cam2'
```

### Cluster Job Submission Guide

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

### Running Your Job

After adjusting the paths in the SLURM script, monitor your job's progress through the SLURM utilities (`squeue`, `sacct`, etc.) and the log files specified in the SBATCH directives.

Make sure all paths and environment names are correctly set to match your project and cluster environment.

The execution calls `/scratch/lgermano/H2O/h2o_ca/data/make_dataset.py` to create and store data in `/scratch/lgermano/H2O/data/raw` or retrieve it, then save it into `/scratch/lgermano/H2O/data/processed`. The entire BEHAVE dataset takes up 4 GB. Choose the labels to train and pick the architecture you want to train in `train_model`. Optionally, you can initialize with old checkpoints.












### Inference

### Example Video 📹

<video width="320" height="240" controls>
  <source src="/scratch/lgermano/H2O/reports/videos/Date02_Sub02_boxsmall_hand_20240117_003809.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# Project Files 📄

This repository contains various files and scripts related to the project. Below is a list of key files and directories along with their descriptions.

## Directories 📁

### trained_models/H2O 🤖
- Description: Directory containing trained models for H2O. 🧠
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: X months ago 📅

## Files 📄

### .gitignore 🚫
- Description: Git ignore file to specify which files and directories should be ignored. 🙈
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 3 months ago 📅

### H2O_a.sh 📜
- Description: Script with added cross attention. 🤖✨
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 2 months ago 📅


## Scripts and Utilities 🛠️

### MLP.py 📜
- Description: Initial commit of MLP script. 🚀
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 5 months ago 📅

## Other Files 📄

### README.md 📚
- Description: README file for the repository. 📖
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: Yesterday 📅


## Model Files 🤖

### model_encoder_only_epoch_4.pt 🧠
- Description: Encoder-only model checkpoint. 📈
- Latest Commit: [Commit ID](https://github.com/jwings1/H2O/tree/2695d4ea13a20c6a675f5a587ee90bb9cbf5e2f1) 🔄
- Last Updated: 2 days ago 📅


## Scripts and Utilities 🛠️


## Citing 📝

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

## Contact Information 📬

For any inquiries, issues, or contributions, please contact:

**Lorenzo Germano**  
- 📧 Email: [lorenzogermano1@outlook.it](mailto:lorenzogermano1@outlook.it)
- 🔗 LinkedIn: [lorenzogermano](https://www.linkedin.com/in/lorenzogermano/)
