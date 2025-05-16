# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file rn50_configs/effnet_32_epochs.yaml \
    --data.train_dataset=train_300_0.1_90.ffcv \
    --data.val_dataset=val_300_0.1_90.ffcv \
    --data.num_workers=32 --data.in_memory=1 \
    --logging.folder='.'
