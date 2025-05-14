cd $SLURM_TMPDIR
mkdir data && cd data
cp /project/def-mpderso/smuralid/ILSVRC2012_img_*.tar .
mkdir train && mv ILSVRC2012_img_train.tar train/
cd train
tar -xf ILSVRC2012_img_train.tar

# Each tar file corresponds to one class
for f in n*.tar; do
    d="${f%.tar}"
    mkdir "$d"
    tar -xf "$f" -C "$d"
    rm "$f"
done
cd ..
mkdir val && mv ILSVRC2012_img_val.tar val/
cd val
tar -xf ILSVRC2012_img_val.tar

# Move files into subdirectories by class using ground truth labels
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
bash valprep.sh
cd $SLURM_TMPDIR/nanodet