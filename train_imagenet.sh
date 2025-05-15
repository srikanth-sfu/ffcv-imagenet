# Set the visible GPUs according to the `world_size` configuration parameter
# Modify `data.in_memory` and `data.num_workers` based on your machine
python train_imagenet.py --config-file rn18_configs/effnet_88_epochs.yaml \
    --data.train_dataset=train_500_0.50_90.ffcv \
    --data.val_dataset=val_500_0.50_90.ffcv \
    --data.num_workers=1 --data.in_memory=1 \
    --logging.folder='.'
