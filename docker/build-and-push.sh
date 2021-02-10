#!/bin/bash -eux

# Run this script from the repo's root folder:
#
# $ ./docker/build-and-push.sh

# 1. Build Docker images for CPU and GPU

image="us-docker.pkg.dev/replicate/raoumer/srrescgan"
cpu_tag="$image:cpu"
gpu_tag="$image:gpu"

docker build -f docker/Dockerfile.cpu --tag "$cpu_tag" .
docker build -f docker/Dockerfile.gpu --tag "$gpu_tag" .

# 2. Test the images on sample data

test_input_folder=/tmp/test-srrescgan/input
mkdir -p $test_input_folder
cp srrescgan_code_demo/samples/bird.png $test_input_folder/
test_output_folder=/tmp/test-srrescgan/output

docker run -it --rm \
    -v $test_input_folder:/code/LR \
    -v $test_output_folder/cpu:/code/sr_results_x4 \
    $cpu_tag

[ -f $test_output_folder/cpu/bird.png ] || exit 1

docker run -it --rm --gpus all \
    -v $test_input_folder:/code/LR \
    -v $test_output_folder/gpu:/code/sr_results_x4 \
    $gpu_tag

[ -f $test_output_folder/gpu/bird.png ] || exit 1

sudo rm -rf "$test_input_folder"
sudo rm -rf "$test_output_folder"

# 3. Push images to Replicate's Docker registry

docker push "$cpu_tag"
docker push "$gpu_tag"
