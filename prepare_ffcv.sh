# Required environmental variables for the script:
export IMAGENET_DIR="${SLURM_TMPDIR}/data"
export WRITE_DIR="."

# Starting in the root of the Git repo:
cd examples;

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
