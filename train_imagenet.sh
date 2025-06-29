# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file rn50_configs/tinynet_tmpl.yaml \
    --data.train_dataset=train_400_0.10_90.ffcv \
    --data.val_dataset=val_400_0.10_90.ffcv \
    --data.num_workers=12 --data.in_memory=1 \
    --model.alpha=$1 --model.beta=$2 \
    --logging.folder='/scratch/smuralid/gridsearch_logs/'
