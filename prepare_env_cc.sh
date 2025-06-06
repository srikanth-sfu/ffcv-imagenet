cd $SLURM_TMPDIR
cp /project/def-mpederso/smuralid/envs/imagenet_trainer.zip .
unzip -qq imagenet_trainer
module load opencv/4.9.0 cuda/12.2
source imagenet_trainer/bin/activate
XDG_CACHE_HOME=$SLURM_TMPDIR pip install numpy==1.25.2
git clone git@github.com:srikanth-sfu/ffcv-imagenet.git
cd ffcv-imagenet
cp /project/def-mpederso/smuralid/*ffcv .
