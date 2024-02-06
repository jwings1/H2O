```markdown
# h2o_ca 🌊

3D Human-Object Interaction in Video 🤖📷

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

### H2O_train.py 📜
- Description: Initial training script. 🚂
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 3 months ago 📅

### H2O_train.sh 📜
- Description: Script with additional features. 🚂✨
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 3 months ago 📅

### [Add more files as needed...] 📁

## Scripts and Utilities 🛠️

### MLP.py 📜
- Description: Initial commit of MLP script. 🚀
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 5 months ago 📅

### MLP.sh 📜
- Description: Initial commit of MLP shell script. 🚀
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 5 months ago 📅

### [Add more scripts and utilities as needed...] 🛠️

## Other Files 📄

### README.md 📚
- Description: README file for the repository. 📖
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: Yesterday 📅

### a.py 📜
- Description: Script with added cross attention. 🤖✨
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 2 months ago 📅

### behave_dataset.py 📜
- Description: Final version of dataset script. 📦✨
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 2 days ago 📅

### [Add more files as needed...] 📄

## Model Files 🤖

### model_encoder_only_epoch_4.pt 🧠
- Description: Encoder-only model checkpoint. 📈
- Latest Commit: [Commit ID](https://github.com/jwings1/H2O/tree/2695d4ea13a20c6a675f5a587ee90bb9cbf5e2f1) 🔄
- Last Updated: 2 days ago 📅

### model_radiant-leaf-3120_epoch_119.pt 🧠
- Description: H2O-CA chain 12 model checkpoint. 📈
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 2 days ago 📅

### [Add more model files as needed...] 🧠

## Scripts and Utilities 🛠️

### robustness_of_distance.py 📜
- Description: Initial commit of distance robustness script. 📊
- Latest Commit: [Commit ID](link to commit) 🔄
- Last Updated: 3 months ago 📅

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

### Build Environment

To create the required environment, use the following command:

```bash
CONDA_OVERRIDE_CUDA=11.7 conda create --name pytcu11 pytorch=2.0.1 pytorch-cuda=11.7 torchvision cudatoolkit=11.7 pytorch-lightning scipy wandb matplotlib --channel pytorch --channel nvidia
```

You can also check the `environment.yml` file located at `/scratch/lgermano/H2O/environment.yml`.

Ensure that your PyTorch and CUDA versions match the compatibility matrix. Refer to [NVIDIA's Dependency Matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) for guidance on compatible versions.

### Training

You can explore certain hyperparameters through a grid search by setting their ranges as flags, as shown in the example:

```bash
sbatch H2O_train6_object.sh --first_option='pose' --second_option='joints' --third_option='obj_pose' --fourth_option='obj_trans' --name='block_cam2'
```

A wrapper is used to communicate with the cluster. The execution calls `/scratch/lgermano/H2O/h2o_ca/data/make_dataset.py` to create and store data in `/scratch/lgermano/H2O/data/raw` or retrieve it, then save it into `/scratch/lgermano/H2O/data/processed`. The entire BEHAVE dataset takes up 4 GB. Choose the labels to train and pick the architecture you want to train in `train_model`. Optionally, you can initialize with old checkpoints.

### Inference

### Example Video (Placeholder)

[Embed your example .mp4 video here]


### Citing

📝 **Citation Template:**

```bibtex
@misc{AuthorLastName_GitHubProjectYear,
  author       = {Author's Last Name},
  title        = {GitHub Project Title},
  year         = {Year},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/AuthorGitHubUsername/GitHubProject}},
  commit       = {GitHubCommitHash},
  doi          = {DOI if applicable},
  note         = {Accessed: Access Date}
}

```


