#!/bin/bash

#SBATCH --job-name="evaluation"
#SBATCH --error=/scratch_net/biwidl307/lgermano/H2O/h2o_ca/log/error/%j.err
#SBATCH --output=/scratch_net/biwidl307/lgermano/H2O/h2o_ca/log/out/%j.out
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:1
##SBATCH --constraint="a6000"


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
cd "${TMPDIR}" || exit 1

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source /itet-stor/lgermano/net_scratch/conda/etc/profile.d/conda.sh

conda activate evaluation

export PYTHONPATH=/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch:$PYTHONPATH
export PYTHONPATH=/scratch_net/biwidl307/lgermano/H2O/h2o_ca:$PYTHONPATH
export PYTHONPATH=/scratch_net/biwidl307/lgermano/quadric-mesh-simplification:$PYTHONPATH
export CONDA_OVERRIDE_CUDA=11.8
export WANDB_DIR=/scratch_net/biwidl307/lgermano/H2O/log/cache

python /scratch_net/biwidl307/lgermano/H2O/h2o_ca/visualizations/predict.py "$@"

echo "DONE!"

echo "Finished at:     $(date)"

exit 0
