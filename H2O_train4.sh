#!/bin/bash

#SBATCH --job-name="rendering"
#SBATCH --error=/scratch_net/biwidl307/lgermano/H2O/log/error/%j.err
#SBATCH --output=/scratch_net/biwidl307/lgermano/H2O/log/out/%j.out
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
##SBATCH --constraint="a6000"

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
cd "${TMPDIR}" || exit 1

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

source /itet-stor/lgermano/net_scratch/conda/etc/profile.d/conda.sh

conda activate render

export PYTHONPATH=/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch:$PYTHONPATH
export CONDA_OVERRIDE_CUDA=11.8

#python /scratch_net/biwidl307/lgermano/H2O/reprojection_human_obj_mesh3D_background.py
python /scratch_net/biwidl307/lgermano/H2O/H2O_train4.py "$@"


echo "DONE!"

echo "Finished at:     $(date)"

exit 0
