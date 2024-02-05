#!/bin/bash

##SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="MLP"
#SBATCH --error=/scratch_net/biwidl307/lgermano/H2O/log/error/%j.err
#SBATCH --output=/scratch_net/biwidl307/lgermano/H2O/log/out/%j.out
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

TMPDIR=/scratch_net/biwidl307/lgermano/H2O/log/cache

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source /itet-stor/lgermano/net_scratch/conda/etc/profile.d/conda.sh
# conda create -n crossvit pytorch torchvision pytorch-cuda --channel pytorch --channel nvidia
# conda activate crossvit
# pip install transformers tensorboard
# cd /scratch_net/biwidl307/lgermano/crossvit

conda activate crossvit

export CONDA_OVERRIDE_CUDA=11.8
export WANDB_API_KEY=34beceb95e6defe789ceb3b540d17ee92a24fd46
export WANDB_DIR=/scratch_net/biwidl307/lgermano/crossvit/wandb
export WANDB_CACHE_DIR=/scratch_net/biwidl307/lgermano/crossvit/wandb/cache


#tensorboard --logdir=logs/ #--host 129.132.67.159
python /scratch_net/biwidl307/lgermano/H2O/MLP_hyperp_tuning.py

echo "DONE!"

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
