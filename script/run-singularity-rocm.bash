#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

tmp=/tmp/$USER/$$
if [[ "$SLURM_TMPDIR" != "" ]]; then
    tmp="$SLURM_TMPDIR/miopen/$$"
fi
mkdir -p $tmp

if [[ "$(hostname -s)" =~ ^g[r,v,a,h] ]]; then nv="--nv"; fi

# if [[ "$(hostname -s)" =~ ^g[m] ]]; then nv="--rocm"; fi

singularity \
    exec $nv \
    --bind $tmp:$HOME/.config/miopen \
    $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
    $(for sqf in /vast/work/public/ml-datasets/imagenet/winter21_whole/*.sqf; do echo "--overlay $sqf:ro"; done) \
    --overlay /scratch/bf996/singularity_containers/openclip_env_rocm.ext3:ro \
    --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_rare_combined.sqf:ro \
    --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_test_set.sqf:ro \
    --overlay /vast/work/public/ml-datasets/bf996/CaptionNet/in100.sqf:ro \
    --overlay /vast/work/public/ml-datasets/bf996/CaptionNet/laion100.sqf:ro \
    --overlay /vast/work/public/ml-datasets/bf996/CaptionNet/openimages1000.sqf:ro \
    --overlay /vast/work/public/ml-datasets/bf996/imagenet-r.sqf:ro \
    --overlay /vast/work/public/ml-datasets/bf996/imagenet-a.sqf:ro \
    --overlay /vast/work/public/ml-datasets/bf996/imagenet-sketch.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
    --overlay /scratch/projects/hegdelab/bf996/datasets/objectnet.sqf:ro \
    --overlay /vast/work/public/ml-datasets/open-images-dataset/open-images-dataset.sqf:ro \
    /scratch/work/public/singularity/rocm5.4.2-ubuntu22.04.2.sif \
    /bin/bash -c "
    source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH:/scratch/bf996/pytorch-image-models
    $args 
"