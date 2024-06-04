#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done
echo $args
ls
if [ "$args" == "" ]; then args="/bin/bash"; fi

if [[ "$(hostname -s)" =~ ^g[r,v,a,h] ]]; then nv="--nv"; fi

singularity \
  exec $nv \
  --overlay /scratch/bf996/singularity_containers/openclip_env_cuda_n.ext3:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_rare_combined.sqf:ro \
  --overlay /scratch/projects/hegdelab/bf996/datasets/arboretum_test_set.sqf:ro \
  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
 source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH:/scratch/bf996/pytorch-image-models
 $args 
"