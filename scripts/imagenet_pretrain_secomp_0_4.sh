#!/bin/bash
#SBATCH --job-name=ptlowerlr
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=def-mpederso
#SBATCH --array=4-7                # example: array indices 0 to 9
#SBATCH --output=/dev/null


cd $SLURM_TMPDIR
cp /scratch/smuralid/imagenet_trainer.zip .
unzip -qq imagenet_trainer
module load opencv/4.9.0 cuda/12.2
source imagenet_trainer/bin/activate
XDG_CACHE_HOME=$SLURM_TMPDIR pip install  --no-deps --no-index numpy==1.25.2 torchinfo
cp -r /scratch/smuralid/ffcv-imagenet .
cd ffcv-imagenet
cp /scratch/smuralid/*ffcv .


bash train_imagenet.sh 1.46 1.45 $SLURM_ARRAY_TASK_ID 
